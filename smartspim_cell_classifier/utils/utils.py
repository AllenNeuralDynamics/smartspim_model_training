import os
import json
import yaml
import logging
from typing import List, Literal, Tuple, Union, overload
import boto3
import s3fs
import tifffile
import numpy as np
import dask.array as da
import natsort

from glob import glob
from pathlib import Path
from imlib.IO.cells import get_cells
from datetime import datetime
from collections import defaultdict
from sklearn.neighbors import KDTree

from ..model import resnet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiff file helpers (replaces cellfinder.core.tools.tiff)
# ---------------------------------------------------------------------------

class TiffFile:
    """A single training cube with one file per channel."""

    def __init__(self, path: str, channels: list, label=None):
        self.path = path
        self.channels = channels
        self.label = label

    @property
    def img_files(self) -> list:
        # e.g. "…Ch0.tif" → strip last 5 chars ("0.tif"), append ch + ".tif"
        return [self.path[:-5] + str(ch) + ".tif" for ch in self.channels]


class TiffList:
    """A pre-built list of channel-0 tiff paths, expanded to TiffFile objects."""

    def __init__(self, ch1_list: list, channels: list, label=None):
        self.ch1_list = natsort.natsorted(ch1_list)
        self.channels = channels
        self.label = label

    def make_tifffile_list(self) -> list:
        files = [
            f for f in self.ch1_list
            if f.lower().endswith("ch" + str(self.channels[0]) + ".tif")
        ]
        return [TiffFile(f, self.channels, self.label) for f in files]


class TiffDir(TiffList):
    """Scans a directory for channel-0 tiff files."""

    def __init__(self, tiff_dir: str, channels: list, label=None):
        ch0_suffix = "ch" + str(channels[0]) + ".tif"
        if tiff_dir.startswith("s3://"):
            fs = s3fs.S3FileSystem(anon=False)
            ch1_list = [
                "s3://" + f
                for f in fs.ls(tiff_dir, detail=False)
                if f.lower().endswith(ch0_suffix)
            ]
        else:
            ch1_list = [
                os.path.join(tiff_dir, f)
                for f in os.listdir(tiff_dir)
                if f.lower().endswith(ch0_suffix)
            ]
        super().__init__(ch1_list, channels, label)


def read_yaml_section(yaml_file, section: str = "data"):
    """
    Read a named section from a YAML file.

    Parameters
    ----------
    yaml_file : str or Path
        Path to the YAML file, or an s3:// URI.
    section : str
        Top-level key to return.

    Returns
    -------
    The contents of the requested section.
    """
    yaml_file = str(yaml_file)
    if yaml_file.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(yaml_file) as f:
            contents = yaml.safe_load(f)
    else:
        with open(yaml_file) as f:
            contents = yaml.safe_load(f)
    return contents[section]


@overload
def make_lists(tiff_files, train: Literal[True] = ...) -> Tuple[List, List, np.ndarray]: ...
@overload
def make_lists(tiff_files, train: Literal[False]) -> Tuple[List, List]: ...
def make_lists(tiff_files, train: bool = True) -> Union[Tuple[List, List, np.ndarray], Tuple[List, List]]:
    """
    Flatten a list of TiffFileList groups into signal, background, and label lists.

    Parameters
    ----------
    tiff_files : list
        Output of ``get_tiff_files`` — a list of TiffFileList groups where
        each image has ``img_files`` (index 0 = signal, 1 = background) and
        a ``label`` attribute ("cell" or "no_cell").
    train : bool
        If True, also return a label array. If False, return only the path lists.

    Returns
    -------
    signal_list : list[Path]
    background_list : list[Path]
    labels : np.ndarray  (only when train=True)
    """
    signal_list = []
    background_list = []
    labels = []

    for group in tiff_files:
        for image in group:
            signal_list.append(image.img_files[0])
            background_list.append(image.img_files[1])
            if train:
                labels.append(0 if image.label == "no_cell" else 1)

    if train:
        return signal_list, background_list, np.array(labels)
    return signal_list, background_list


def load_json(fname):
    '''
    Function to load JSON files

    Parameters:
    -----------

    fname: str
        pathway to the json file

    Returns:
    --------
    data: dict
        information from json formated as a dictionary
    '''

    with open(fname, 'r') as f:
        data = json.load(f)

    return data

def parse_annotation_dict(data, layer_name, dims):
    '''
    Function that takes a dictionary loaded from a Neuroglancer
    JSON and retrieves the location of points from an annotation
    layer

    Parameters:
    -----------
    data: dict
        The data that comes from loading a neuroglancer JSON
    layer_name: str
        The name of the annotation layer you are parsing. Is the
        same as the tab name of the layer when viewed in Neuroglancer

    Returns:
    --------
    annot_list: list
        A list of the points from the specificed annotation layer

    '''

    layers = data['layers']
    logger.debug(f"Volume dims: {dims}")

    for layer in layers:
        if layer['type'] == 'annotation' and layer['name'].lower() == layer_name.lower():
            annotations = layer['annotations']
    
    annot_list = []
    for point in annotations:
        loc = point['point']

        if int(round(loc[0])) > 0 and int(round(loc[0])) < dims[0]:
            if int(round(loc[1])) > 0 and int(round(loc[1])) < dims[1]:
                if int(round(loc[2])) > 0 and int(round(loc[2])) < dims[2]:
                    annot_list.append(
                        [
                            int(round(loc[0])),
                            int(round(loc[1])),
                            int(round(loc[2])),
                        ]
                    )
    
    return annot_list

def read_json_as_dict(filepath: str):
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary

def get_crop_size(img_res, crop_microns):
    """
    Get the size of the crop around a cell in pixel values

    Parameters:
    -----------
    img_res: list[float]
        A list containing the resolution of the 0 level zarr in um/px
    
    crop_microns: list[float]
        A list containing the size of training crops in um

    Returns:
    --------
    scale: list[int]
        A list with the number of pixels to pad on each side of the
        crop from the center of a cell
    """

    scales = []
    for res, micron in zip(img_res, crop_microns):
        scale = np.ceil(micron/(res*2)).astype(int)
        scales.append(scale)
    return scales

def cells_to_list(cells):
    '''
    Convert cellfinder Cell objects to list
    
    Parameters:
    -----------
    cells: list[Cells]
        list of cell objects from xml
    
    Return:
    -------
    cell_list: list
        list of cell locations
    '''

    cell_list = []

    for cell in cells:
        if cell.type == 2:
            cell_list.append(
                [
                    cell.z,
                    cell.y,
                    cell.x,
                ]
            )
    
    return cell_list

def get_metrics(annot_list, class_list):
    '''
    Calculate metrics on a set of annotation points.

    Parameters:
    -----------
    annot_list: list
        annotations locations taken from Neuroglancer JSON
    class_list: list
        classified cell locations taken from XML output of pipeline

    Returns:
    --------
    metrics: df
        a compressed dataframe with classification metrics
    cell_data: list
        a list with information on how each annotation and classified
        cell where categorized
    '''

    c_loc = np.array(class_list)
    b_loc = np.array(annot_list)

    if len(c_loc) == 0:
        metrics = {
            'Total_Annotations': len(b_loc),
            'Total_Classifications': len(c_loc),
            'TP': 0.0,
            'FP': 0.0,
            'FN': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
        }

        cell_data = []
        for b, loc in enumerate(b_loc):
            cell_data.append([b, loc, np.nan, np.nan, 0, 0, 1]) 

        return metrics, cell_data
    else:
        tree = KDTree(c_loc, leaf_size = 2)
        out, dist = tree.query_radius(
            b_loc,
            7, 
            return_distance=True,
            sort_results=True
        )
    
        data = []
        cell_data = []
        false_negative = 0
        true_positive = 0

        for b, o in enumerate(out):
            if len(o) > 0: 
                found = False
                for idx in o:
                    if idx not in data:
                        true_positive += 1
                        cell_data.append([b, b_loc[b], idx, c_loc[idx], 1, 0, 0])
                        data.extend([idx])
                        found = True
                        break
                if not found:
                    false_negative += 1
                    cell_data.append([b, b_loc[b], np.nan, np.nan, 0, 0, 1])    
            else:
                false_negative += 1
                cell_data.append([b, b_loc[b], np.nan, np.nan, 0, 0, 1])

        # get false positives                         
        for c in range(len(c_loc)):
            if c not in data:
                cell_data.append([np.nan, np.nan, c, c_loc[c], 0, 1, 0])
    
        false_positive = abs(len(c_loc) - true_positive)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    
        metrics = {
            'Total_Annotations': len(b_loc),
            'Total_Classifications': len(c_loc),
            'TP': true_positive,
            'FP': false_positive,
            'FN': false_negative,
            'Precision': precision,
            'Recall': recall,
            'F1': 2 * ((precision * recall) / (precision + recall)),
        }
    
        return metrics, cell_data

def get_client():
    '''
    Creates an instance of a s3 client for interacting with AWS

    Returns
    -------
    client.s3
        S3 object of botocore.client module

    '''
    return boto3.client('s3')

def fetch_zarrs_aws(lt_list, bucket = 'aind-open-data', level = 0):
    """
    get Zarrs from AWS

    Parameters:
    -----------
    lt_list: list
        List of the dataset identifiers 
    bucket: str Optional
        aws bucket where data is located. Default = 'aind-open-data'
    level: int Optional
        the zarr level you want for classification

    Returns:
    --------
    data: defaultdict
        Dictionary with data parsed by dataset and channel
    """
    
    s3_client = get_client()

    paginator = s3_client.get_paginator('list_objects')
    result = paginator.paginate(Bucket = bucket, Delimiter='/')
    channel_options = [
        'Ex_445_Em_469',
        'Ex_488_Em_525',
        'Ex_561_Em_593', 
        'Ex_561_Em_600',
        'Ex_639_Em_660',
        'Ex_639_Em_667',
        'Ex_639_Em_680',
        'Ex_647_Em_690',
        ]
    
    data = defaultdict(dict)

    for prefix in result.search('CommonPrefixes'):
        file = prefix.get('Prefix')
        if all(x in file for x in ['SmartSPIM', 'stitched']):
            lt_id = file.split("_")[1]
            if lt_id in lt_list:
                lt_id = int(lt_id)
                
                for channel in channel_options:
                    try:
                        zarr_link = os.path.join(
                            's3://',
                            bucket,
                            file,
                            'image_tile_fusing',
                            'OMEZarr', 
                            f"{channel}.zarr"
                        )
                        zarr_data = da.from_zarr(zarr_link, level).squeeze()
                        data[lt_id][channel] = zarr_data
                    except:
                        pass
                    
                    try:
                        zarr_link = os.path.join(
                            's3://',
                            bucket,
                            file,
                            'processed',
                            'stitching',
                            'OMEZarr', 
                            f"{channel}.zarr"
                        )
                        zarr_data = da.from_zarr(zarr_link, level).squeeze()
                        data[lt_id][channel] = zarr_data
                    except:
                        pass

                    
    return data

def build_model(save_path, model_params):
    """
    Create ResNet model

    Parameters:
    -----------

    model_path: str
        Where you want to save the model
    model_params:
        Parameters for the new model

    Returns:
    --------
    model_name: str
        name of new model

    """

    model = resnet.build_improved_model(**model_params)

    date = datetime.now()
    date = f"{date.year}_{date.month}_{date.day}"

    layers = model_params['network_depth'].replace("-", "_")
    model_name = f"{date}_smartspim_resnet_{layers}.keras"

    model.summary(print_fn=logger.info)
    logger.info(f"Built model: {model_name}")

    return model, model_name

def save_yaml_file(save_path, lt_id, channel):
    """
    Save the yaml file required for trainiong the network.
    Format for file was pulled from cellfinder-napari github

    Parameters:
    -----------
    save_path: Pathlike
        Where you want to save the yaml file
    lt_id: str
        The identification number for a given dataset
    channel: str
        The channel wavelength of the signal being annotatated

    Returns:
    --------
    None
    """

    output_directory = os.path.join(save_path, 'ymls')

    if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    yaml_filename = os.path.join(output_directory, f"training_{lt_id}_{channel}.yml")
    yaml_section = [
        {
            "cube_dir": os.path.join(output_directory, lt_id, channel, "cells"),
            "cell_def": "",
            "type": "cell",
            "signal_channel": 0,
            "bg_channel": 1,
        },
        {
            "cube_dir": os.path.join(output_directory, lt_id, channel, "non_cells"),
            "cell_def": "",
            "type": "no_cell",
            "signal_channel": 0,
            "bg_channel": 1,
        },
    ]

    yaml_contents = {"data": yaml_section}
    with open(yaml_filename, "w") as outfile:
        yaml.dump(
            yaml_contents, outfile, default_flow_style=False
        )

    return

def save_image_patches(
    lt_id,
    channel,
    save_path, 
    signal, 
    bkg, 
    cells_dict, 
    crop_pad, 
    level
):
    """
    Save image patches for each annotated cell separated by 
    true positive and false positives

    Parameters:
    -----------
    lt_id: str
        The identification number for a given dataset
    channel: str
        The channel wavelength of the signal being annotatated
    save_path: Pathlike
        root to where the image patches will be saved
    signal: dask.array
        volume of the channel where cells were annotated
    bkg: dask.array
        volume of backgroud channel containing only autofluorescence
    cells_dict: dict
        the locations of the cells imported form neuroglancer json
    crop_pad: list
        The number of pixels along each dimension to pad each side of
        cell centroid
    level: int
        The level of the zarr that is being used for classification

    Returns:
    --------
    None
    """

    for key, cells in cells_dict.items():
        logger.info(f"Saving patches for class '{key}': {len(cells)} cells")
        set_folder = os.path.join(save_path, lt_id, channel, key)
        if not os.path.exists(set_folder):
            os.makedirs(set_folder)
       
        for cell in cells:
        
            cell = [int(round(loc / 2**int(level))) for loc in cell]

            indecies = [(int(c-p), int(c+p)) for c, p in zip(cell, crop_pad)]
            sig_patch = signal[
                indecies[0][0]:indecies[0][1],
                indecies[1][0]:indecies[1][1],
                indecies[2][0]:indecies[2][1]
            ]
            bkg_patch = bkg[
                indecies[0][0]:indecies[0][1],
                indecies[1][0]:indecies[1][1],
                indecies[2][0]:indecies[2][1]
            ]
        
            sig_fname = f"pCellz{cell[0]}y{cell[1]}x{cell[2]}Ch0.tif"
            bkg_fname = f"pCellz{cell[0]}y{cell[1]}x{cell[2]}Ch1.tif"

            tifffile.imwrite(os.path.join(set_folder, sig_fname), sig_patch)
            tifffile.imwrite(os.path.join(set_folder, bkg_fname), bkg_patch)
    
    save_yaml_file(save_path, lt_id, channel)

    return

def get_annotations(annot_path, cells_layer, noncells_layer):
    """
    Get the true positives and false positives for ground truth cells

    Parameters
    ------------------------
    annot_path: Pathlike
        The location of the annotated neuroglancer data

    Returns
    ------------------------
    annotations: dict
        disctionary with cell locations broken up by type
    """

    if 'json' in annot_path:
        pass
    else:
        annot_path = glob(os.path.join(annot_path, "*.json"))[0]

    annot_dict = read_json_as_dict(annot_path)
    annot_layers = annot_dict['layers']

    with open('../code/config_files/crop_configs.yml', "r") as stream:
        configs = yaml.safe_load(stream)

    annotations = {}
    name = annot_path.split('/')[-1][:-5]

    logger.info(f"Parsing annotations from: {name}")

    for layer in annot_layers:
        if layer['type'] == 'annotation' and layer['name'].lower() in [cells_layer.lower(), noncells_layer.lower()]:
            annot_list = []
            logger.info(f"Annotation layer: {layer['name'].lower()}, total count: {len(layer['annotations'])}")
            for annotation in layer['annotations']:
                if ('TH' in name) or ('sparse' in name):
                    offset = configs['data'][name[7:]]['region']
                    annot_list.append(
                        [
                            int(annotation['point'][0]) + offset[0],
                            int(annotation['point'][1]) + offset[1],
                            int(annotation['point'][2]) + offset[2]
                        ]
                    )
                else:
                    annot_list.append(
                        [
                            int(annotation['point'][0]),
                            int(annotation['point'][1]),
                            int(annotation['point'][2])
                        ]
                    )
            
            if cells_layer.lower() == layer['name'].lower():
                annotations['cells'] = annot_list
            elif noncells_layer.lower() == layer['name'].lower():
                annotations['non_cells'] = annot_list
    
    return annotations

def downsample_training_set(
    path,
    downsample,
    save_path,
):

    files = glob(os.path.join(path, "*.tif"), recursive = True)

    for file in files:
        fname = os.path.basename(file)
        input_dir = os.path.dirname(file)
        output_dir = input_dir.replace('data', 'results')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        


    return

def build_training_set(
    level,
    annotation_path,
    cells_layer,
    noncells_layer,
    resolution,
    crop_size,
    save_path,
):
    """
    Collects data required for building a new dataset

    Parameters:
    -----------
    level: int
        The level of the zarr that is being used for classification
    annotation path:
        The location of the JSON files containing neuroglancer annotations
    resolution: list
        highest resolution of the Zarr the annotations where collected from
    crop_size: list
        size of crop around each cell in microns
    save_path: Pathlike
        root to where the image patches will be saved
    """

    dataset_files = glob(os.path.join(annotation_path, "*.json"))
    lt_list = [os.path.basename(file).split('_')[0] for file in dataset_files]

    logger.info(f"Found {len(lt_list)} datasets for processing")

    aws_links = fetch_zarrs_aws(lt_list, level=level)
    crop_px = get_crop_size(resolution, crop_size)

    logger.info(f"Retrieved zarrs for {len(aws_links)} dataset(s) from AWS")

    for file in dataset_files:

        cells = get_annotations(file, cells_layer, noncells_layer)

        data_info = os.path.basename(file).split('_')
        lt_id, channel = data_info[0], data_info[1]
        lt_dict = aws_links[int(lt_id)]
        logger.info(f"Processing dataset: {lt_id}, signal channel: {channel}")
        for k, v in lt_dict.items():
            if channel in k:
                logger.info(f"Found signal channel {k} for {lt_id}")
                sig_array = v
            elif any(ch in k for ch in ['639', '647']):
                logger.info(f"Found background channel {k} for {lt_id}")
                bkg_array = v

        save_image_patches(
            lt_id=lt_id,
            channel=channel,
            signal=sig_array,
            bkg=bkg_array,
            cells_dict=cells,
            crop_pad=crop_px,
            save_path=save_path,
            level=level,
        )

        del sig_array, bkg_array

        logger.info(f"Saved patches for dataset: {file}")
    
    return