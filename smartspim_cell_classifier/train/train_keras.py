import logging
import numpy as np
import random
import keras
import json
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split

from ..utils.utils import (
    TiffDir,
    read_yaml_section,
    make_lists
)
from ..utils.volume_augment import customImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable()
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_pred = K.cast(y_pred, "float32")
    y_true = K.cast(y_true, "float32")

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    return K.mean(f1)


def parse_yaml(yaml_files, section="data"):
    data = []
    for yaml_file in yaml_files:
        data.extend(read_yaml_section(yaml_file, section))
    return data


def get_tiff_files(yaml_contents):

    tiff_lists = []
    for d in yaml_contents:
        if d["bg_channel"] < 0:
            channels = [d["signal_channel"]]
        else:
            channels = [d["signal_channel"], d["bg_channel"]]
        tiff_lists.append(TiffDir(d["cube_dir"], channels, d["type"]))

    tiff_files = [tiff_dir.make_tifffile_list() for tiff_dir in tiff_lists]
    return tiff_files


def _derive_cube_dir_root(yml_file):
    """Return the training-set root directory from a yml file path.

    Works for both local paths and s3:// URIs. The yml is always at
    <root>/ymls/<name>.yml so the root is two levels up.
    """
    parts = str(yml_file).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[:-2])


def balance_dataset(signal, background, labels):

    tp_count = sum(labels)
    fp_count = len(labels) - tp_count
    difference = tp_count - fp_count

    logger.info(f"There are {tp_count} true positives and {fp_count} false positives")

    if difference > 0:
        if difference > fp_count:
            logger.info(f"Resampling all false positives to add {fp_count} data points")
            idx = [c for c, i in enumerate(labels) if i == 0]
            samples = random.sample(idx, len(idx))
        else:
            logger.info(f"Resampling false positives to add {difference} data points")
            idx = [c for c, i in enumerate(labels) if i == 0]
            samples = random.sample(idx, difference)
    elif difference < 0:
        if abs(difference) > tp_count:
            logger.info(f"Resampling all true positives to add {tp_count} data points")
            idx = [c for c, i in enumerate(labels) if i == 1]
            samples = random.sample(idx, len(idx))
        else:
            logger.info(f"Resampling true positives to add {abs(difference)} data points")
            idx = [c for c, i in enumerate(labels) if i == 1]
            samples = random.sample(idx, abs(difference))
    else:
        logger.info("Dataset is already balanced")
        return signal, background, labels

    new_signal, new_background, new_labels = [], [], []
    for i in samples:
        new_signal.append(signal[i])
        new_background.append(background[i])
        new_labels.append(labels[i])

    signal = np.concatenate((signal, np.array(new_signal)))
    background = np.concatenate((background, np.array(new_background)))
    labels = np.concatenate((labels, np.array(new_labels)))

    tp_count = sum(labels)
    fp_count = len(labels) - tp_count

    logger.info(
        f"Balanced dataset: {tp_count} true positives, {fp_count} false positives"
    )

    return signal, background, labels


def generate_cubes(signal, background, shape):

    images = np.empty(
        ((len(signal),) + shape + (2,)),
        dtype=np.float32,
    )

    for idx, (signal_path, bkg_path) in enumerate(zip(signal, background)):
        signal_img = np.moveaxis(imread(signal_path), 0, 2)
        bkg_img = np.moveaxis(imread(bkg_path), 0, 2)

        images[idx, :, :, :, 0] = signal_img
        images[idx, :, :, :, 1] = bkg_img

    return images


def _flow(signal, background, labels, shape, gen, batch_size, predict, shuffle):
    for b in gen.flow(
        signal=signal,
        background=background,
        labels=labels,
        points=None,
        shape=shape,
        batch_size=batch_size,
        predict=predict,
        shuffle=shuffle,
    ):
        yield (b[0], b[1])


def build_data_generators(yaml_files, augment_params, model_params, override_cube_dir=False):
    """
    Load the dataset from yaml files and return train/validation generators.

    This is intentionally decoupled from training so the generators can be
    used directly in custom training loops.

    Parameters
    ----------
    yaml_files : list[Path]
        Paths to the cellfinder-format yaml files describing the cube dataset.
    augment_params : dict
        Keyword arguments forwarded to ``customImageDataGenerator``.
    model_params : dict
        Must contain ``balance``, ``test_fraction``, and ``batch_size``.
    override_cube_dir : bool
        When True, ignore the ``cube_dir`` stored in each yml and rebuild it
        from the yml file's location (``yml_path/training_set``), keeping the
        last 3 path components of the original value (``lt_id/channel/type``).

    Returns
    -------
    training_gen : generator
        Yields ``(batch_images, batch_labels)`` for training.
    validation_gen : generator
        Yields ``(batch_images, batch_labels)`` for validation.
    epoch_size : int
        Number of training samples — use for ``steps_per_epoch``.
    n_val : int
        Number of validation samples — use for ``validation_steps``.
    """
    logger.info(f"Parsing {len(yaml_files)} yaml file(s)")
    yaml_contents = []
    for yaml_file in yaml_files:
        entries = read_yaml_section(yaml_file)
        if override_cube_dir:
            root = _derive_cube_dir_root(yaml_file)
            for entry in entries:
                trailing = "/".join(str(entry["cube_dir"]).replace("\\", "/").rstrip("/").split("/")[-3:])
                entry["cube_dir"] = f"{root}/{trailing}"
        yaml_contents.extend(entries)
    tiff_files = get_tiff_files(yaml_contents)

    signal_train, background_train, labels_train = make_lists(tiff_files)
    logger.info(f"Loaded {len(signal_train)} total samples from disk")

    if model_params['balance']:
        signal_train, background_train, labels_train = balance_dataset(
            signal_train, background_train, labels_train
        )
    else:
        tp_count = sum(labels_train)
        fp_count = len(labels_train) - tp_count
        logger.info(f"There are {tp_count} true positives and {fp_count} false positives")

    if model_params['test_fraction'] > 0:
        logger.info("Splitting data into training and validation datasets")
        (
            signal_train,
            signal_test,
            background_train,
            background_test,
            labels_train,
            labels_test,
        ) = train_test_split(
            signal_train,
            background_train,
            labels_train,
            test_size=model_params['test_fraction'],
        )
        logger.info(
            f"Train samples: {len(signal_train)}, validation samples: {len(signal_test)}"
        )

    cube_shape = tuple([14, 14, 26])
    batch_size = model_params['batch_size']

    gen_train = customImageDataGenerator(**augment_params)
    gen_val = customImageDataGenerator(
        featurewise_center=augment_params['featurewise_center'],
        featurewise_std_normalization=augment_params['featurewise_std_normalization'],
        samplewise_center=augment_params['samplewise_center'],
        samplewise_std_normalization=augment_params['samplewise_std_normalization'],
        percentile_normalization=augment_params['percentile_normalization'],
        percentile_range=augment_params['percentile_range'],
        zca_whitening=augment_params['zca_whitening'],
        zca_epsilon=1e-7,
    )

    if augment_params['featurewise_center'] or augment_params['featurewise_std_normalization']:
        logger.info("Fitting featurewise statistics on training data")
        max_samples = augment_params.get('featurewise_max_samples', None)
        gen_train.fit(signal_train, background_train, shape=cube_shape, inference=False, max_samples=max_samples)
        gen_val.fit(signal_train, background_train, shape=cube_shape, inference=False, max_samples=max_samples)

    training_gen = _flow(
        signal=signal_train,
        background=background_train,
        labels=labels_train,
        gen=gen_train,
        shape=cube_shape,
        batch_size=batch_size,
        predict=False,
        shuffle=True,
    )

    validation_gen = _flow(
        signal=signal_test,
        background=background_test,
        labels=labels_test,
        gen=gen_val,
        shape=cube_shape,
        batch_size=batch_size,
        predict=False,
        shuffle=True,
    )

    return training_gen, validation_gen, len(signal_train), len(signal_test)



def train(model, training_gen, validation_gen, model_params, epoch_size, n_val):
    """
    Compile, fit, and save the model.

    Parameters
    ----------
    model : keras.Model
        Compiled or uncompiled Keras model.
    training_gen : generator
        Training data generator (e.g. from ``build_data_generators``).
    validation_gen : generator
        Validation data generator.
    model_params : dict
        Must contain ``schedule``, ``monitor``, ``batch_size``, ``epochs``,
        and ``output_dir``.
    epoch_size : int
        Total number of training samples.
    n_val : int
        Total number of validation samples.

    Returns
    -------
    history : keras.callbacks.History
    """
    batch_size = model_params['batch_size']
    batch_num = np.ceil(epoch_size / batch_size)

    callbacks = []
    if model_params['schedule'] == 'Cosine':
        lr = 1e-3
        first_decay = int(batch_num) * 8
        alpha = 0.1

        logger.info(
            f"Cosine schedule: lr={lr}, first_decay_steps={first_decay}, alpha={alpha}"
        )

        schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=first_decay,
            t_mul=2.0,
            m_mul=1.0,
            alpha=alpha,
            name="SGDRDecay",
        )

        callbacks.append(
            ModelCheckpoint(
                str(model_params['output_dir'] / "best_model_epoch_{epoch:03d}_val_metric_{val_recall_at_p90:.4f}.keras"),
                monitor=model_params['monitor'],
                save_best_only=True,
                mode='max',
            )
        )

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=schedule))

    elif model_params['schedule'] == 'ReduceLR':
        logger.info("Using ReduceLROnPlateau schedule with EarlyStopping")
        callbacks.append(
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            )
        )

    logger.info(
        f"Beginning training: {model_params['epochs']} epochs, "
        f"batch_size={batch_size}, steps_per_epoch={int(np.ceil(epoch_size / batch_size))}"
    )

    history = model.fit(
        training_gen,
        steps_per_epoch=np.ceil(epoch_size / batch_size).astype(int),
        validation_data=validation_gen,
        validation_steps=np.ceil(n_val / batch_size).astype(int),
        epochs=model_params['epochs'],
        callbacks=callbacks,
    )

    logger.info(f"Saving model to {model_params['output_dir']}")
    model.save(model_params['output_dir'] / "model.keras")
    model.save_weights(model_params['output_dir'] / "model.weights.h5")

    config = {
        'shape': (14, 14, 26, 2),
        'network_depth': "18-layer",
        'learning_rate': 0.0003,
        'starting_features': 64,
        'dropout_rate': 0.3,
        'axis': 3,
        'use_focal_loss': True,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'use_attention': True,
        'attention_kernel_size': 5,
        'l2_reg': 1e-4,
        'optimizer': 'adam',
        'number_classes': 1,
        'classification_activation': 'sigmoid',
        'multiscale_reduction': 1,
        'normalization': 'per_channel_percentile',
        'percentile_range': [1, 99],
    }

    with open(model_params['output_dir'] / "model_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training complete. Metrics recorded: {list(history.history.keys())}")

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'test'], loc='upper left')

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'test'], loc='upper left')
    plt.tight_layout()

    fig.savefig('/results/model_performance.png')

    return history


def run(model_params, augment_params, model):
    """Convenience wrapper: build generators then train."""
    training_gen, validation_gen, epoch_size, n_val = build_data_generators(
        model_params['yaml_file'], augment_params, model_params,
        override_cube_dir=model_params.get('override_cube_dir', False),
    )
    return train(model, training_gen, validation_gen, model_params, epoch_size, n_val)
