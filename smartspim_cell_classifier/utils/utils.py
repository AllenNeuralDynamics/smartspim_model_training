import os
import json
import logging

import numpy as np
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)


def load_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def parse_annotation_dict(data, layer_name, dims):
    layers = data['layers']
    logger.debug(f"Volume dims: {dims}")

    annotations = []
    for layer in layers:
        if layer['type'] == 'annotation' and layer['name'].lower() == layer_name.lower():
            annotations = layer['annotations']
            break

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
    dictionary = {}
    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)
    return dictionary


def cells_to_list(cells):
    cell_list = []
    for cell in cells:
        if cell.type == 2:
            cell_list.append([cell.z, cell.y, cell.x])
    return cell_list


def get_metrics(annot_list, class_list):
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
        tree = KDTree(c_loc, leaf_size=2)
        out, dist = tree.query_radius(b_loc, 7, return_distance=True, sort_results=True)

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

        for c in range(len(c_loc)):
            if c not in data:
                cell_data.append([np.nan, np.nan, c, c_loc[c], 0, 1, 0])

        false_positive = abs(len(c_loc) - true_positive)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics = {
            'Total_Annotations': len(b_loc),
            'Total_Classifications': len(c_loc),
            'TP': true_positive,
            'FP': false_positive,
            'FN': false_negative,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        }

        return metrics, cell_data
