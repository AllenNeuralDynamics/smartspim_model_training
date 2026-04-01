import threading

import numpy as np
import keras.backend as K


class customImageDataGenerator(object):
    """
    Data loader for SmartSPIM cell classification inference.

    Handles patch extraction from 3D zarr volumes and percentile
    normalization. Augmentation is not applied during inference.

    Parameters
    ----------
    percentile_normalization : bool
        If True, normalize each patch to [0, 1] using percentile clipping.
    percentile_range : tuple of (float, float)
        Lower and upper percentile bounds used for normalization.
    """

    def __init__(
        self,
        percentile_normalization=False,
        percentile_range=(1, 99),
    ):
        self.percentile_normalization = percentile_normalization
        self.percentile_range = percentile_range
        self.ordered_points = []

    def flow(self, signal, background, points, shape,
             batch_size=32, shuffle=False, seed=None):
        """
        Return a PatchIterator that extracts patches centered on each point.

        Parameters
        ----------
        signal : np.ndarray
            3D signal volume (z, y, x).
        background : np.ndarray
            3D background volume, same shape as signal.
        points : list of [x, y, z]
            Cell centroid coordinates.
        shape : tuple of (x, y, z)
            Patch size to extract around each centroid.
        batch_size : int
        shuffle : bool
        seed : int or None
        """
        return PatchIterator(
            image_data_generator=self,
            signal=signal,
            background=background,
            points=points,
            shape=shape,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def _apply_percentile_normalization(self, x):
        """
        Normalize a single-channel patch to [0, 1] using percentile clipping.

        Parameters
        ----------
        x : np.ndarray
            Input patch array (H, W, D).

        Returns
        -------
        np.ndarray
            Normalized patch in range [0, 1].
        """
        x = x.astype(np.float32)
        p_low, p_high = np.percentile(x, self.percentile_range)

        if p_high > p_low:
            x_norm = (x - p_low) / (p_high - p_low)
            x_norm = np.clip(x_norm, 0, 1)
        else:
            x_norm = np.ones_like(x) * 0.5

        return x_norm

    def standardize(self, x):
        """
        Apply normalization to a single-channel patch.

        Parameters
        ----------
        x : np.ndarray
            Input patch.

        Returns
        -------
        np.ndarray
            Normalized patch.
        """
        if self.percentile_normalization:
            x = self._apply_percentile_normalization(x)
        return x


class Iterator(object):
    """Abstract base class for data iterators."""

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while True:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)
            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class PatchIterator(Iterator):
    """
    Iterator that extracts image patches from 3D numpy volumes.

    Patches are centered on the provided cell coordinates, normalized
    via the parent generator's `standardize` method, and returned as
    batches of shape (batch_size, x, y, z, 2).
    """

    def __init__(self, points, signal, background, image_data_generator,
                 batch_size=32, shape=None, shuffle=False, seed=None):

        self.image_data_generator = image_data_generator
        self.signal = signal
        self.background = background
        self.points = points
        self.shape = shape

        for pt in points:
            self.image_data_generator.ordered_points.append(pt)

        super(PatchIterator, self).__init__(len(points), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_images = np.empty(
            ((current_batch_size,) + self.shape + (2,)),
            dtype=np.float32,
        )

        for i, j in enumerate(index_array):
            loc = self.points[j]

            x = self.signal[
                loc[2] - self.shape[0] // 2: loc[2] + self.shape[0] // 2,
                loc[1] - self.shape[1] // 2: loc[1] + self.shape[1] // 2,
                loc[0] - self.shape[2] // 2: loc[0] + self.shape[2] // 2
            ]
            x = self.image_data_generator.standardize(x.astype(K.floatx()))
            batch_images[i, :, :, :, 0] = x

            y = self.background[
                loc[2] - self.shape[0] // 2: loc[2] + self.shape[0] // 2,
                loc[1] - self.shape[1] // 2: loc[1] + self.shape[1] // 2,
                loc[0] - self.shape[2] // 2: loc[0] + self.shape[2] // 2
            ]
            y = self.image_data_generator.standardize(y.astype(K.floatx()))
            batch_images[i, :, :, :, 1] = y

        return batch_images
