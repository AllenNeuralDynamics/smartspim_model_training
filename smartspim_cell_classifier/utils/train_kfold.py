"""
Copy and pasted: cellfinder_core.train.train_yml script here
the function hard codes the shape of augmented cubes causing an error with
the cube size we use. Didn't feel like creating new repo. easiest fix
"""

import os
import random
import numpy as np
import tensorflow as tf
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    ArgumentTypeError,
)
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal

from brainglobe_utils.general.numerical import (
    check_positive_float,
    check_positive_int,
)
from brainglobe_utils.general.system import ensure_directory_exists
from brainglobe_utils.IO.cells import find_relevant_tiffs
from brainglobe_utils.IO.yaml import read_yaml_section
from fancylog import fancylog
from sklearn.model_selection import train_test_split, KFold

import cellfinder_core as program_for_log

tf_suppress_log_messages = [
    "sample_weight modes were coerced from",
    "multiprocessing can interact badly with TensorFlow",
]

depth_type = Literal["18", "34", "50", "101", "152"]
layer_type = Literal["18-layer", "34-layer", "50-layer", "101-layer", "152-layer"]

models: Dict[depth_type, layer_type] = {
    "18": "18-layer",
    "34": "34-layer",
    "50": "50-layer",
    "101": "101-layer",
    "152": "152-layer",
}


def valid_model_depth(depth):
    """
    Ensures a correct existing_model is chosen
    :param value: Input value
    :param models: Dict of allowed models
    :return: Input value, if it corresponds to a valid existing_model
    """

    if depth in models.keys():
        return depth
    else:
        raise ArgumentTypeError(
            f"Model depth: {depth} is not valid. Please "
            f"choose one of: {list(models.keys())}"
        )


def misc_parse(parser):
    misc_parser = parser.add_argument_group("Misc options")
    misc_parser.add_argument(
        "--n-free-cpus",
        dest="n_free_cpus",
        type=check_positive_int,
        default=2,
        help="The number of CPU cores on the machine to leave "
        "unused by the program to spare resources.",
    )
    misc_parser.add_argument(
        "--max-ram",
        dest="max_ram",
        type=check_positive_float,
        default=None,
        help="Maximum amount of RAM to use (in GB) - not currently fully "
        "implemented for all parts of cellfinder",
    )
    misc_parser.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        help="Save .csv files of cell locations (in addition to xml)."
        "Useful for importing into other software.",
    )
    misc_parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Debug mode. Will increase verbosity of logging and save all "
        "intermediate files for diagnosis of software issues.",
    )
    misc_parser.add_argument(
        "--sort-input-file",
        dest="sort_input_file",
        action="store_true",
        help="If set to true, the input text file will be sorted using "
        "natural sorting. This means that the file paths will be "
        "sorted as would be expected by a human and "
        "not purely alphabetically",
    )
    return parser


def training_parse():
    from cellfinder_core.download.cli import (
        download_directory_parser,
        model_parser,
    )

    training_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    training_parser.add_argument(
        "-y",
        "--yaml",
        dest="yaml_file",
        nargs="+",
        required=True,
        type=str,
        help="The path to the yaml run file.",
    )
    training_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        required=True,
        type=str,
        help="Output directory for the final model.",
    )
    training_parser.add_argument(
        "--continue-training",
        dest="continue_training",
        action="store_true",
        help="Continue training from an existing trained model. If no model "
        "or model weights are specified, this will continue from the "
        "included model.",
    )
    training_parser.add_argument(
        "--trained-model",
        dest="trained_model",
        type=str,
        help="Path to the trained model",
    )
    training_parser.add_argument(
        "--model-weights",
        dest="model_weights",
        type=str,
        help="Path to existing model weights",
    )
    training_parser.add_argument(
        "--network-depth",
        dest="network_depth",
        type=valid_model_depth,
        default="50",
        help="Resnet depth (based on He et al. (2015)",
    )
    training_parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=check_positive_int,
        default=16,
        help="Training batch size",
    )
    training_parser.add_argument(
        "--epochs",
        dest="epochs",
        type=check_positive_int,
        default=100,
        help="Number of training epochs",
    )
    training_parser.add_argument(
        "--test-fraction",
        dest="test_fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation",
    )
    training_parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=check_positive_float,
        default=0.0001,
        help="Learning rate for training the model",
    )
    training_parser.add_argument(
        "--no-augment",
        dest="no_augment",
        action="store_true",
        help="Don't apply data augmentation",
    )
    training_parser.add_argument(
        "--save-weights",
        dest="save_weights",
        action="store_true",
        help="Only store the model weights, and not the full model. Useful to "
        "save storage space.",
    )
    training_parser.add_argument(
        "--no-save-checkpoints",
        dest="no_save_checkpoints",
        action="store_true",
        help="Store the model at intermediate points during training",
    )
    training_parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Log to output_directory/tensorboard",
    )
    training_parser.add_argument(
        "--save-progress",
        dest="save_progress",
        action="store_true",
        help="Save training progress to a .csv file",
    )

    training_parser = misc_parse(training_parser)
    training_parser = model_parser(training_parser)
    training_parser = download_directory_parser(training_parser)
    args = training_parser.parse_args()

    return args

def parse_yaml(yaml_files, section="data"):
    data = []
    for yaml_file in yaml_files:
        data.extend(read_yaml_section(yaml_file, section))
    return data

def get_tiff_files(yaml_contents):
    from cellfinder_core.tools.tiff import TiffDir, TiffList

    tiff_lists = []
    for d in yaml_contents:
        if d["bg_channel"] < 0:
            channels = [d["signal_channel"]]
        else:
            channels = [d["signal_channel"], d["bg_channel"]]
        if "cell_def" in d and d["cell_def"]:
            ch1_tiffs = [
                os.path.join(d["cube_dir"], f)
                for f in os.listdir(d["cube_dir"])
                if f.endswith("Ch" + str(channels[0]) + ".tif")
            ]
            tiff_lists.append(
                TiffList(
                    find_relevant_tiffs(ch1_tiffs, d["cell_def"]),
                    channels,
                    d["type"],
                )
            )
        else:
            tiff_lists.append(TiffDir(d["cube_dir"], channels, d["type"]))

    tiff_files = [tiff_dir.make_tifffile_list() for tiff_dir in tiff_lists]
    return tiff_files


def cli():
    args = training_parse()
    ensure_directory_exists(args.output_dir)

    fancylog.start_logging(
        args.output_dir,
        program_for_log,
        variables=[args],
        log_header="CELLFINDER TRAINING LOG",
    )

    output_dir = Path(args.output_dir)
    run(
        output_dir,
        args.yaml_file,
        n_free_cpus=args.n_free_cpus,
        trained_model=args.trained_model,
        model_weights=args.model_weights,
        install_path=args.install_path,
        model=args.model,
        network_depth=args.network_depth,
        learning_rate=args.learning_rate,
        continue_training=args.continue_training,
        test_fraction=args.test_fraction,
        batch_size=args.batch_size,
        no_augment=args.no_augment,
        tensorboard=args.tensorboard,
        save_weights=args.save_weights,
        no_save_checkpoints=args.no_save_checkpoints,
        save_progress=args.save_progress,
        epochs=args.epochs,
    )


def run(
    output_dir,
    yaml_file,
    shape,
    n_free_cpus=2,
    trained_model=None,
    model_weights=None,
    install_path='/data/',
    model="resnet50_tv",
    network_depth="50",
    learning_rate=0.0001,
    continue_training=False,
    test_fraction=0.1,
    batch_size=16,
    no_augment=False,
    tensorboard=False,
    save_weights=False,
    no_save_checkpoints=True,
    save_progress=False,
    epochs=100,
    balance = False,
):
    from cellfinder_core.main import suppress_tf_logging

    suppress_tf_logging(tf_suppress_log_messages)

    from tensorflow.keras.callbacks import (
        CSVLogger,
        ModelCheckpoint,
        TensorBoard,
    )

    from cellfinder_core.classify.cube_generator import CubeGeneratorFromDisk
    from cellfinder_core.classify.tools import get_model, make_lists
    from cellfinder_core.tools.prep import prep_training

    start_time = datetime.now()

    ensure_directory_exists(output_dir)

    yaml_contents = parse_yaml(yaml_file)

    tiff_files = get_tiff_files(yaml_contents)

    signal_train, background_train, labels_train = make_lists(tiff_files)
    cells = np.where(np.array(labels_train) == 1)[0]
    non_cells = np.where(np.array(labels_train) == 0)[0]
    print(f"Dataset Size: {len(signal_train)}\n Balance: {len(cells)} cells and {len(non_cells)} non-cells")

    if balance:
        if len(cells) != len(non_cells):

            if len(cells) > len(non_cells):

                keep = random.sample(list(cells), len(non_cells))
                keep_idx = np.concatenate([keep, non_cells])

            elif len(cells) < len(non_cells):

                keep = random.sample(list(non_cells), len(cells))
                keep_idx = np.concatenate([keep, cells])

            signal_train = [signal_train[x] for x in keep_idx]
            background_train = [background_train[x] for x in keep_idx]
            labels_train = [labels_train[x] for x in keep_idx]
            print(f"Balanced Dataset: {sum(labels_train)} cells and {len(labels_train) - sum(labels_train)} non-cells")

    kf = KFold(n_splits=5)
    fold = 0
    for train_index, test_index in kf.split(signal_train):
        print(f"Running fold {fold} of 5 for Kfold validation:")

        model = tf.keras.models.load_model(install_path)

        sig_train = [signal_train[x] for x in train_index]
        sig_test =  [signal_train[x] for x in test_index]
        bkg_train = [background_train[x] for x in train_index]
        bkg_test =  [background_train[x] for x in test_index]
        l_train = [labels_train[x] for x in train_index]
        l_test =  [labels_train[x] for x in test_index]

        validation_generator = CubeGeneratorFromDisk(
            sig_test,
            bkg_test,
            shape=shape,
            labels=l_test,
            batch_size=batch_size,
            train=True,
        )

        # for saving checkpoints
        base_checkpoint_file_name = "-epoch.{epoch:02d}-loss-{val_loss:.3f}.h5"

        training_generator = CubeGeneratorFromDisk(
            sig_train,
            bkg_train,
            shape=shape,
            labels=l_train,
            batch_size=batch_size,
            shuffle=True,
            train=True,
            augment=not no_augment,
        )

        callbacks = []

        if tensorboard:
            logdir = output_dir / "tensorboard"
            ensure_directory_exists(logdir)
            tensorboard = TensorBoard(
                log_dir=logdir,
                histogram_freq=0,
                write_graph=True,
                update_freq="epoch",
            )
            callbacks.append(tensorboard)

        if not no_save_checkpoints:
            if save_weights:
                filepath = str(output_dir / ("weight" + base_checkpoint_file_name))
            else:
                filepath = str(output_dir / ("model" + base_checkpoint_file_name))

            checkpoints = ModelCheckpoint(
                filepath,
                save_weights_only=save_weights,
            )
            callbacks.append(checkpoints)

        if save_progress:
            filepath = str(output_dir / f"training_fold_{fold}.csv")
            csv_logger = CSVLogger(filepath)
            callbacks.append(csv_logger)


        model.fit(
            training_generator,
            validation_data=validation_generator,
            use_multiprocessing=False,
            epochs=epochs,
            callbacks=callbacks,
        )

        if save_weights:
            model.save_weights(str(output_dir / "model_weights.h5"))
        else:
            model.save(output_dir / f"model_fold_{fold}.h5")
        
        fold += 1

if __name__ == "__main__":
    cli()