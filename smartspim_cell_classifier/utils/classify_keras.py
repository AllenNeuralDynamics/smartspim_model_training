import os
import gc
import yaml
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zarr
import time
import tifffile
import random
import keras
import dask.array as da
import numpy as np

from glob import glob
from pathlib import Path
import keras.backend as K
from cellfinder.core.classify import classify
from cellfinder.core.classify.cube_generator import CubeGeneratorFromFile
from cellfinder.core.classify.tools import get_model
from imlib.IO.cells import save_cells, get_cells
from imlib.cells.cells import Cell

from .volume_augment import customImageDataGenerator

logger = logging.getLogger(__name__)

models = {
    "18": "18-layer",
    "34": "34-layer",
    "50": "50-layer",
    "101": "101-layer",
    "152": "152-layer",
}

def npy_to_cells(data, padding, dims):
    '''
    Convert the .npy file from detection to cellfinder Cells for classification

    Parameters
    ----------
    data: ArrayLike
        cell x corrdinate array from the detection

    Returns
    -------
    cells: list
        list of Cell objects
    '''
    
    cell_locs = np.load(data)
    
    cells = []
    for cell in cell_locs:

        cell = [loc-padding for loc in cell]
        
        if all(loc > 0 for loc in cell) & all(loc <= dim for loc, dim in zip(cell, dims)):
        
            cells.append(
                Cell(
                    (cell[2], cell[1], cell[0]),
                    1
                )
            )

    return cells

def get_yaml_config(filename):
    """
    Get default configuration from a YAML file.

    Parameters
    ------------------------
    filename: str
        String where the YAML file is located.

    Returns
    ------------------------
    Dict
        Dictionary with the configuration
    """

    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)

    return config

class Classification():

    def __init__(self, params):
        self.cells = params['cells']
        self.signal_path = params['signal_path']
        self.background_path = params['background_path']
        self.input_scale = params['level']
        self.offsets = params['offsets']
        self.model = params['model']
        self.network_depth= params['network_depth']
        self.batch_size = params['batch_size']
        self.save_path = params['save_path']
        self.cond = params['cond']
        self.detect_pad = params['detect_pad']
        self.pad = params['pad']
        self.test = params['test']
        self.means = params['means']
        self.stds = params['stds']
        self.rescale= params['rescale']

    def calculate_offsets(self, blocks, chunk_size):
        """
        creates list of offsets for each chunk based on its location
        in the dask array

        Parameters
        ----------
        blocks : tuple
            The number of blocks in each direction (z, col, row)

        chunk_size : tuple
            The number of values along each dimention of a chunk (z, col, row)

        Return
        ------
        offests: list
            The offsets of each block in "C order
        """
        offsets = []
        for dv in range(blocks[0]):
            for ap in range(blocks[1]):
                for ml in range(blocks[2]):
                    offsets.append(
                        [
                            chunk_size[2] * ml,
                            chunk_size[1] * ap,
                            chunk_size[0] * dv,
                        ]
                    )
        return offsets

    def load_zarr(self, path):
        data = da.from_zarr(path)
        return data[0, 0, :, :, :]

    def get_dims(self):
        dims = [
            self.offsets[3]-self.offsets[0],
            self.offsets[4]-self.offsets[1],
            self.offsets[5]-self.offsets[2]
        ]
        return dims

    def scale_cells(self):

        new_cells = []

        for cell in self.cells:
            x = int(cell.x / 2**int(self.input_scale)) + self.pad
            y = int(cell.y / 2**int(self.input_scale)) + self.pad
            z = int(cell.z / 2**int(self.input_scale)) + self.pad

            cell.x = x
            cell.y = y
            cell.z = z
            
            new_cells.append(cell)


        return new_cells

    def smartspim_class_config(self):
        home = Path.home()
        model_type = "resnet50_tv"
        install_path = '/scratch/'
        create_model = False
        model_path = self.model
        depth = "18"

        logger.info(f"Loading model from: {model_path}")
        return {
            'trained_model': model_path,
            'model_weights': None,
            'n_free_cpus': 15,
            'batch_size': 32,
            'voxel_sizes': [4, 3.6, 3.6],
            'network_voxel_sizes': [4, 3.6, 3.6],
            'cube_width': 14,                
            'cube_height': 14,
            'cube_depth': 26,
            'network_depth': depth
        }
    

    def get_block_cells(self, offsets, chunk, padding):
        "Remove cells in the padded region that are < 0 or > range"
        cells = []
        for cell in self.cells:
            x = cell.x / 2
            y = cell.y / 2
            z = cell.z / 2
            if (x > offsets[0]) and (x < (offsets[0] + chunk)):
                if (y > offsets[1]) and (y < (offsets[1] + chunk)):
                    if (z > offsets[2]) and (z < (offsets[2] + chunk)):
                        cell.x = int(round(x - offsets[0] + padding))
                        cell.y = int(round(y - offsets[1] + padding))
                        cell.z = int(round(z - offsets[2] + padding))
                        cells.append(cell)
        return np.array(cells)

    def generator(self, signal, background, labels, points, shape, gen, batch_size, predict, shuffle):

        for b in gen.flow(signal=signal, background=background, points=points, labels=labels, shape=shape, batch_size = batch_size, predict=predict, shuffle=shuffle):
            yield (b[0], b[1])

    def inference_generator(self, signal, background, labels, points, shape, gen, batch_size, predict, shuffle):

        for b in gen.flow(signal=signal, background=background, points=points, labels=labels, shape=shape, batch_size = batch_size, predict=predict, shuffle=shuffle):
            yield (b,)

    def run_model(self, cells, padding, signal, background, smartspim_config, offset, block = 0):


        gen = customImageDataGenerator(percentile_normalization = True, percentile_range=(1,99))


        cell_list = []
        for cell in cells:
            cell_list.append(
                [cell.x, cell.y, cell.z]
            )

        logger.info(f"Running inference on {len(cell_list)} cells")

        inference_generator = self.inference_generator(
            signal=signal,
            background=background,
            labels=None,
            points=cell_list,
            gen=gen,
            shape=tuple([14, 14, 26]),
            batch_size=self.batch_size,
            predict=True,
            shuffle=False
        )


        model = keras.saving.load_model(self.model)
        predictions = model.predict(
            inference_generator,
            steps=len(cell_list)//self.batch_size + 1,
            verbose=True,
            callbacks=None,
        )

        logger.debug(f"Raw predictions shape: {predictions.shape}")
        predictions[:, 1] -=0.05
        predictions[:, 0] +=0.05

        predictions = predictions.round()
        predictions = predictions.astype("uint16")

        predictions = np.argmax(predictions, axis=1)
        classified_cells = []

        # only go through the "extractable" points
        logger.info(f"Extractable points after generator filtering: {len(gen.ordered_points)}")
        for idx, cell in enumerate(gen.ordered_points):
            new_cell = Cell(
                pos = {
                    'x': cell[0], 
                    'y': cell[1],
                    'z': cell[2],
                
                },
                cell_type = 1

            )

            new_cell.type = predictions[idx] + 1
            classified_cells.append(new_cell)


        cells_out = []
        for cell in classified_cells:
            x = (cell.x + offset[0] - padding) * 2
            y = (cell.y + offset[1] - padding) * 2
            z = (cell.z + offset[2] - padding) * 2
            cell.x = x
            cell.y = y
            cell.z = z
            
            cells_out.append(cell)


        save_cells(
            cells=cells_out, 
            xml_file_path=os.path.join(self.save_path, "classifications", f"{self.cond}_classified_cells_{block}.xml"),            
        )

    def run(self):

        signal_path = os.path.join(self.signal_path, self.input_scale)  
        bkg_path = os.path.join(self.background_path, self.input_scale) 

        logger.info(f"Signal path: {signal_path}")
        logger.info(f"Background path: {bkg_path}")

        smartspim_config = self.smartspim_class_config()
        logger.info(f"Predicting with model parameters: {smartspim_config}")
        
        signal_zarr = self.load_zarr(signal_path)

        bkg_zarr = self.load_zarr(bkg_path)
        
        if not self.test:
            self.offsets = [int(offset / 2**int(self.input_scale)) for offset in self.offsets]

            signal = np.asarray(
                signal_zarr[
                    self.offsets[0] - self.pad:self.offsets[3] + self.pad,
                    self.offsets[1] - self.pad:self.offsets[4] + self.pad,
                    self.offsets[2] - self.pad:self.offsets[5] + self.pad
                ]
            )

            background = np.asarray(
                bkg_zarr[
                    self.offsets[0] - self.pad:self.offsets[3] + self.pad,
                    self.offsets[1] - self.pad:self.offsets[4] + self.pad,
                    self.offsets[2] - self.pad:self.offsets[5] + self.pad
                ]
            )

            cells = self.scale_cells()

            self.run_model(cells, self.pad, signal, background, smartspim_config, offset = [0, 0, 0], block = "")


        else:
            signal = signal_zarr
            background = bkg_zarr

            signal = signal.rechunk((256, 256, 256))
            background = background.rechunk((256, 256, 256))

            blocks = signal.to_delayed().ravel()
            bkg_blocks = background.to_delayed().ravel()
            block_offsets = self.calculate_offsets(signal.numblocks, signal.chunksize)
            counts = range(len(blocks))

            padding = (
                int(
                    np.ceil((smartspim_config['cube_depth'] + 1) / 2)
                ),
                int(
                    np.ceil((smartspim_config['cube_depth'] + 1) / 2)
                )
            )

            for sig, bkg, offset, count in zip(blocks, bkg_blocks, block_offsets, counts):
                

                block_cells = self.get_block_cells(offset, chunk = 256, padding = padding[0])
                
                
                if len(block_cells) == 0:
                    continue

                model = get_model(
                    existing_model=smartspim_config['trained_model'],
                    model_weights=None,
                    network_depth=models[smartspim_config['network_depth']],
                    inference=True,
                )

                signal = np.pad(
                    sig.compute(),
                    padding,
                    mode="reflect"
                )

                background = np.pad(
                    bkg.compute(), 
                    padding,
                    mode="reflect"
                )

                self.run_model(block_cells, padding[0], model, signal, background, smartspim_config, offset, count)

                del model
                K.clear_session()
                gc.collect()

            all_cells = []
            for cell_path in glob(os.path.join('../results/classifications', '*.xml')):
                
                cells = get_cells(cell_path)
                for cell in cells:
                    all_cells.append(cell)
                
            save_cells(all_cells, f'../results/classifications/{self.cond}_all_classifications.xml')

        return