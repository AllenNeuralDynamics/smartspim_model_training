import os
import yaml
import logging

import numpy as np
import pandas as pd

from glob import glob
from .utils import cells_to_list, get_metrics, load_json, parse_annotation_dict
from imlib.IO.cells import get_cells, save_cells

logger = logging.getLogger(__name__)

def get_all_cells(cell_path, channel, region, cond):

    folder = cell_path.split('/')[2]
    cell_path = f'/data/{folder}/image_cell_segmentation/{channel}/detected_cells.xml'
    cells = get_cells(cell_path)    
    fname = f"../results/detections/{cond}_proposals.xml"

    save_cells(cells, fname)

    return cells

def get_region_cells(cell_path, channel, region, cond):

    folder = cell_path.split('/')[2]
    cell_path = f'/data/{folder}/image_cell_segmentation/{channel}/detected_cells.xml'
    cells_xml = get_cells(cell_path)
    
    cells = []
    for cell in cells_xml:
        if (cell.z > region[0]) and cell.z < region[3]:
            if (cell.y > region[1]) and cell.y < region[4]:
                if (cell.x > region[2]) and cell.x < region[5]:
                    cell.z -= region[0]
                    cell.y -= region[1]
                    cell.x -= region[2]
                    cells.append(cell)
    
    fname = f"../results/detections/{cond}_proposals.xml"

    save_cells(cells, fname)

    return cells

def get_volume_dimensions(configs):

    dim_dict = {}
    for dataset, data in configs['data'].items():

        dim_dict[dataset] = [
            data['region'][3] - data['region'][0],
            data['region'][4] - data['region'][1],
            data['region'][5] - data['region'][2],
        ]

    return dim_dict

def run(params):

    annotation_paths = glob(
        os.path.join(params['benchmark_json_path'], '*.json')
    )

    with open('../code/config_files/crop_configs.yml', "r") as stream:
        configs = yaml.safe_load(stream)

    dim_dict = get_volume_dimensions(configs)

    logger.info(f"Found {len(annotation_paths)} annotation JSON(s): {annotation_paths}")

    layer_name = 'cells' # which annotation layer you are pulling
    annotation_dict = {}
    for ap in annotation_paths:
        info = os.path.basename(ap)[:-5].split("_")
        dataset = '_'.join(info[1:3])
        logger.info(f"Parsing annotations: {ap}")

        json_data = load_json(ap)
        annotations = parse_annotation_dict(json_data, params['layer_name'], dim_dict[dataset])
        annotation_dict[f"{dataset}"] = annotations
        logger.info(f"Found {len(annotations)} annotations in layer '{layer_name}'")

    metric_list = []
    for dataset in glob(os.path.join(params['classified_xml_path'], '*.xml')):

        cond = os.path.basename(dataset).split('_')
        cond = '_'.join(cond[:2])

        logger.info(f"Computing metrics for condition: {cond}")
    
        annotations = annotation_dict[cond]

        class_cells = get_cells(dataset)
        classifications = cells_to_list(class_cells)
    
        metrics, data = get_metrics(annotations, classifications)
        metrics['Condition'] = cond
        metric_list.append(metrics)
        
    metric_df = pd.DataFrame(metric_list)
    metric_df = metric_df.sort_values(by = 'Condition')
    metric_df.to_excel('../results/metrics.xlsx')

if __name__ == "__main__":
    run()