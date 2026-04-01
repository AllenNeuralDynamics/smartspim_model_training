import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import dask.array as da
import numpy as np
import pandas as pd

from imlib.IO.cells import save_cells
from imlib.cells.cells import Cell

from ..DataLoader.customDataloader import customImageDataGenerator

logger = logging.getLogger(__name__)


class Classification():

    def __init__(self, params):
        self.cells = params['cells']
        self.signal_path = params['signal_path']
        self.background_path = params['background_path']
        self.input_scale = params['level']
        self.offsets = params['offsets']
        self.model_path = params['model_path']
        self.batch_size = params['batch_size']
        self.save_path = params['save_path']
        self.cond = params['cond']
        self.pad = params['pad']
        self.percentile_normalization = params.get('percentile_normalization', False)
        self.percentile_range = params.get('percentile_range', (1, 99))

    def load_zarr(self, path):
        data = da.from_zarr(path, storage_options={"anon": True})
        return data[0, 0, :, :, :]

    def scale_cells(self):
        new_cells = []
        for cell in self.cells:
            new_cell = Cell(
                pos={
                    'x': int(cell.x / 2**int(self.input_scale)) + self.pad,
                    'y': int(cell.y / 2**int(self.input_scale)) + self.pad,
                    'z': int(cell.z / 2**int(self.input_scale)) + self.pad,
                },
                cell_type=cell.type,
            )
            new_cells.append(new_cell)
        return new_cells

    def inference_generator(self, signal, background, points, shape, gen, batch_size, shuffle):
        for b in gen.flow(signal=signal, background=background, points=points,
                          shape=shape, batch_size=batch_size, shuffle=shuffle):
            yield (b,)

    def run_model(self, cells, padding, signal, background, offset):

        gen = customImageDataGenerator(
            percentile_normalization=self.percentile_normalization,
            percentile_range=self.percentile_range,
        )

        cell_list = [[cell.x, cell.y, cell.z] for cell in cells]

        logger.info(f"Running inference on {len(cell_list)} cells")

        inference_gen = self.inference_generator(
            signal=signal,
            background=background,
            points=cell_list,
            gen=gen,
            shape=(14, 14, 26),
            batch_size=self.batch_size,
            shuffle=False,
        )

        model = keras.saving.load_model(self.model_path)
        predictions = model.predict(
            inference_gen,
            steps=len(cell_list) // self.batch_size + 1,
            verbose=True,
            callbacks=None,
        )

        logger.debug(f"Raw predictions shape: {predictions.shape}")

        raw_probs = predictions[:, 1]
        predictions = np.argmax(predictions.round().astype("uint16"), axis=1)

        classified_cells = []
        logger.info(f"Extractable points after generator filtering: {len(gen.ordered_points)}")
        for idx, cell in enumerate(gen.ordered_points):
            new_cell = Cell(pos={'x': cell[0], 'y': cell[1], 'z': cell[2]}, cell_type=1)
            new_cell.type = predictions[idx] + 1
            classified_cells.append(new_cell)

        cells_out = []
        prob_records = []
        for idx, cell in enumerate(classified_cells):
            cell.x = (cell.x + offset[0] - padding) * 2
            cell.y = (cell.y + offset[1] - padding) * 2
            cell.z = (cell.z + offset[2] - padding) * 2
            cells_out.append(cell)
            prob_records.append([cell.z, cell.y, cell.x, raw_probs[idx]])

        pd.DataFrame(prob_records, columns=["z", "y", "x", "prob"]).to_csv(
            os.path.join(self.save_path, "classifications", f"{self.cond}_probabilities.csv"),
            index=False,
        )

        save_cells(
            cells=cells_out,
            xml_file_path=os.path.join(self.save_path, "classifications", f"{self.cond}_classified_cells.xml"),
        )

    def run(self):
        signal_path = f"{self.signal_path}/{self.input_scale}"
        bkg_path = f"{self.background_path}/{self.input_scale}"

        logger.info(f"Signal path: {signal_path}")
        logger.info(f"Background path: {bkg_path}")

        signal_zarr = self.load_zarr(signal_path)
        bkg_zarr = self.load_zarr(bkg_path)

        self.offsets = [int(offset / 2**int(self.input_scale)) for offset in self.offsets]

        signal = np.asarray(signal_zarr[
            self.offsets[0] - self.pad:self.offsets[3] + self.pad,
            self.offsets[1] - self.pad:self.offsets[4] + self.pad,
            self.offsets[2] - self.pad:self.offsets[5] + self.pad,
        ])

        background = np.asarray(bkg_zarr[
            self.offsets[0] - self.pad:self.offsets[3] + self.pad,
            self.offsets[1] - self.pad:self.offsets[4] + self.pad,
            self.offsets[2] - self.pad:self.offsets[5] + self.pad,
        ])

        cells = self.scale_cells()
        self.run_model(cells, self.pad, signal, background, offset=[0, 0, 0])
