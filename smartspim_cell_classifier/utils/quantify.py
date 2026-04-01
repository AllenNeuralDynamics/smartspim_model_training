import os
import yaml
import logging

import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from glob import glob
from .utils import cells_to_list, get_metrics, load_json, parse_annotation_dict
from .plots import run_all_plots
from imlib.IO.cells import get_cells

logger = logging.getLogger(__name__)


def get_all_cells(zarr_url, channel, region, cond, save_path, bucket_name):
    """
    Download detected cells XML from S3 and return as a list of Cell objects.

    Parameters
    ----------
    zarr_url : str
        S3 URL to the zarr directory, e.g.
        's3://aind-open-data/SmartSPIM_XYZ.../image_tile_fusing/OMEZarr/'
    channel : str
        Channel name, e.g. 'Ex_488_Em_525'
    region : list
        Region offsets (unused here, kept for API compatibility).
    cond : str
        Condition label used for the output filename.
    save_path : str
        Root output directory; XML is saved to {save_path}/detections/.
    bucket_name : str
        S3 bucket name, e.g. 'aind-open-data'.
    """
    dataset_folder = zarr_url.split('/')[3]
    s3_key = f"{dataset_folder}/image_cell_segmentation/{channel}/detected_cells.xml"
    local_path = os.path.join(save_path, "detections", f"{cond}_detected_cells.xml")

    logger.info(f"Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket_name, s3_key, local_path)

    return get_cells(local_path)


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

    with open(params['benchmark_config_path'], "r") as stream:
        configs = yaml.safe_load(stream)

    dim_dict = get_volume_dimensions(configs)

    logger.info(f"Found {len(annotation_paths)} annotation JSON(s): {annotation_paths}")

    annotation_dict = {}
    for ap in annotation_paths:
        info = os.path.basename(ap)[:-5].split("_")
        dataset = '_'.join(info[1:3])
        logger.info(f"Parsing annotations: {ap}")

        json_data = load_json(ap)
        annotations = parse_annotation_dict(json_data, params['layer_name'], dim_dict[dataset])
        annotation_dict[dataset] = annotations
        logger.info(f"Found {len(annotations)} annotations in layer '{params['layer_name']}'")

    metric_list = []
    for xml_path in glob(os.path.join(params['classified_xml_path'], '*.xml')):
        cond = os.path.basename(xml_path).split('_')
        cond = '_'.join(cond[:2])

        logger.info(f"Computing metrics for condition: {cond}")

        if cond not in annotation_dict:
            logger.warning(f"No annotations found for condition '{cond}', skipping.")
            continue

        annotations = annotation_dict[cond]
        class_cells = get_cells(xml_path)
        classifications = cells_to_list(class_cells)

        metrics, _ = get_metrics(annotations, classifications)
        metrics['Condition'] = cond
        metric_list.append(metrics)

    metric_df = pd.DataFrame(metric_list)
    metric_df = metric_df.sort_values(by='Condition')
    metric_df.to_excel(os.path.join(params['save_path'], 'metrics.xlsx'))

    plots_path = os.path.join(params['save_path'], 'plots')
    run_all_plots(plots_path, metric_df=metric_df)
