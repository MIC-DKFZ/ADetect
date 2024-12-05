"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

import argparse
from utilities.data_io import load_segmentation, valid_image_format, partition_list
from utilities.segmentation_volumetry import calculate_segmentation_volumes
import pandas as pd
import json
import os
from multiprocessing import Pool


def process_folder_mp(folder: str, label_dict: dict, num_processes: int, diseased: bool) -> dict:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Segfolder is not valid: {folder}.")

    segfiles = [os.path.join(folder, file) for file in os.listdir(folder) if valid_image_format(file)]
    if not len(segfiles):
        raise ValueError(f"No valid segmentations found in folder: {folder}.")

    with Pool(num_processes) as pool:
        inputs = partition_list(segfiles, num_processes)
        fct_args = [(item, label_dict, diseased) for item in inputs]
        results = pool.starmap(process_folder, fct_args)
        results_joined = {}
        for result in results:
            results_joined = {**results_joined, **result}
        return results_joined


def process_folder(files: list, label_dict: dict, diseased: bool) -> dict:
    output = {}
    for file in files:
        seg = load_segmentation(file)
        print(f"Calculating segmentation volumes for file: {file}...")
        volumes = calculate_segmentation_volumes(seg)

        try:
            verify_measurements(volumes, label_dict)
        except ValueError as e:
            raise ValueError(f"Invalid labels encountered in segmentation file: {file}.") from e

        converted = convert_measurements(volumes, label_dict)
        converted['is_AD'] = diseased
        output[file] = converted
    return output


def convert_measurements(measurements: dict, label_dict: dict) -> dict:
    converted = {}
    for label, structure in label_dict.items():
        if label in measurements.keys():
            converted[structure] = measurements[label]['volume_ml']
        else:
            converted[structure] = 0
    return converted


def verify_measurements(measurements: dict, label_dict: dict):
    labels_measurements = set(measurements.keys())
    labels_ref = set(label_dict.keys())
    if not labels_measurements.issubset(labels_ref):
        raise ValueError(f"Unexpected labels encountered in volume measurements. "
                         f"Unexpected labels: {labels_measurements - labels_ref}.")


def verify_labels(ref_set: set, label_dict: dict):
    columns = ref_set - {'is_AD'}
    values = set(label_dict.values())
    if columns != values:
        raise ValueError(f"Invalid labels in label file. Missing expected labels: {values-columns}. "
                         f"Unexpected labels encountered: {columns-values}.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segfolder-diseased', type=str, required=True,
                        help='Path to a directory of nifti segmentation files for AD cases.')
    parser.add_argument('--segfolder-healthy', type=str, required=True,
                        help='Path to a directory of nifti segmentation files for non-AD cases.')
    parser.add_argument('--outfile', type=str, required=True,
                        help='Path to the output csv file where the volume measurements will be written.')
    parser.add_argument('--labelfile', type=str, required=True,
                        help='Path to a json file of label-to-structure mappings. '
                             'Different models can have different mappings, due to label prioritization '
                             'for the membrane.')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of processes for parallelization.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    segfolder_diseased = args.segfolder_diseased
    segfolder_healthy = args.segfolder_healthy
    label_file = args.labelfile
    outfile = args.outfile
    num_processes = args.num_processes

    try:
        with open(label_file, 'r') as f:
            label_dict = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not read label information from file: {label_file}.") from e

    ref_set = {'false_lumen_ascending', 'membrane', 'false_lumen_descending', 'hemopericardium',
               'aortic wall haematoma', 'false lumen in brachiocephalic trunk', 'carotid artery right',
               'subclavian artery right', 'carotid artery left', 'subclavian artery left', 'is_AD'}

    try:
        verify_labels(ref_set, label_dict)
    except ValueError as e:
        raise ValueError(f"Unexpected labels in label file: {label_file}.") from e

    output_diseased = process_folder_mp(segfolder_diseased, label_dict, num_processes, diseased=True)
    output_healthy = process_folder_mp(segfolder_healthy, label_dict, num_processes, diseased=False)

    df_diseased = pd.DataFrame.from_dict(output_diseased, orient='index')
    df_healthy = pd.DataFrame.from_dict(output_healthy, orient='index')

    print(f"Writing output...")
    try:
        pd.concat([df_diseased, df_healthy], axis=0).to_csv(outfile)
    except Exception as e:
        raise RuntimeError(f"Could not write dataframe to csv in file: {outfile}.") from e

    print("All finished.")


if __name__ == '__main__':
    main()
