"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import json
import pandas as pd
import os
import SimpleITK as sitk


def valid_image_format(file: str):
    return file.endswith('.nii') or file.endswith('.nii.gz') or file.endswith('.nrrd') or file.endswith('.nhdr')


def load_segmentations(segfolder: str) -> dict:
    if not os.path.isdir(segfolder):
        raise FileNotFoundError(f"{segfolder}: no valid directory.")

    files = [file for file in os.listdir(segfolder) if valid_image_format(file)]
    if not len(files):
        raise ValueError(f"No valid segmentation files found in folder: {segfolder}. "
                         f"Valid filetypes are: .nii, .nii.gz, .nrrd, .nhdr.")

    print(f"Found n={len(files)} segmentations in folder: {segfolder}.")
    segmentations = {}
    for file in files:
        try:
            print(f"Reading file: {file}...")
            seg = sitk.ReadImage(os.path.join(segfolder, file))
        except Exception as e:
            raise RuntimeError(f"Could not read segmentation from file: {os.path.join(segfolder, file)}.") from e

        segmentations[file] = seg

    return segmentations


def load_segmentation(segfile: str):
    if not os.path.isfile(segfile):
        raise FileNotFoundError(f"{segfile}: no valid segmentation file.")

    try:
        print(f"Reading file: {segfile}...")
        seg = sitk.ReadImage(segfile)
    except Exception as e:
        raise RuntimeError(f"Could not read segmentation from file: {segfile}.") from e

    return seg


def merge_seg_data_and_gt(seg_data: pd.DataFrame, seg_data_gt: pd.DataFrame):
    if seg_data.shape != seg_data_gt.shape:
        raise ValueError(f"Segmentation input must have equal formatting for test and ground truth data. "
                         f"Shape for test data: {seg_data.shape}, "
                         f"shape for ground truth data: {seg_data_gt.shape}")

    seg_data_indices = {os.path.basename(index) for index in set(seg_data.index)}
    gt_indices = {os.path.basename(index) for index in set(seg_data_gt.index)}

    if not seg_data_indices == gt_indices:
        raise ValueError(f"Ids differ between input data and ground truth data. "
                         f"Extra ids in volume measurements: {seg_data_indices-gt_indices}. "
                         f"Extra ids in gt measurements: {gt_indices-seg_data_indices}.")

    index_mapping = {}
    for i, index in enumerate(seg_data_gt.index):
        for index2 in seg_data.index:
            if os.path.basename(index) == os.path.basename(index2):
                index_mapping[index] = os.path.join(os.path.dirname(index2), os.path.basename(index))

    seg_data_gt_reindexed = seg_data_gt['false_lumen_ascending'].rename(index=index_mapping)

    seg_data['false_lumen_ascending_gt'] = seg_data_gt_reindexed


def read_segmentation_input(segfile: str):
    try:
        seg_data = pd.read_csv(segfile, index_col=0)
        return seg_data
    except Exception as e:
        raise RuntimeError(f"Could not read volume measurements from file: {segfile}.") from e


def check_valid_input(seg_data: pd.DataFrame):
    segdata_check_both_classes(seg_data)
    segdata_check_nan(seg_data)


def segdata_check_both_classes(seg_data: pd.DataFrame):
    missing_values = {0, 1} - set(seg_data['is_AD'].unique())
    if missing_values:
        raise ValueError(f"Class labels for both AD- and non-AD cases must be provided. "
                         f"Label value {missing_values} has not been provided.")


def segdata_check_nan(seg_data: pd.DataFrame):
    missing_values = seg_data.isna()
    if missing_values.any().any():
        missing_positions = missing_values.stack()
        missing_positions = missing_positions[missing_positions]
        raise ValueError(f"Error: Segmentation data contains NA values. Missing values at: {missing_positions}.")


def write_results(result: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(result, f, indent=4, default=custom_serializer)


def custom_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy ndarray to list
    elif isinstance(obj, dict):
        return {key: custom_serializer(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [custom_serializer(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)  # Convert numpy int64 to Python int
    raise TypeError("Type not serializable")


def partition_list(input_list: list, num_parts: int):
    if not len(input_list):
        raise ValueError(f"Cannot partition empty list.")
    if num_parts > len(input_list):
        raise ValueError(f"More parts (n={num_parts}) requested than list has elements (m={len(input_list)}).")

    n = len(input_list)
    chunk_size = n // num_parts
    remainder = n % num_parts

    partitions = []
    start = 0

    for i in range(num_parts):
        end = start + chunk_size + (1 if i < remainder else 0)
        partitions.append(input_list[start:end])
        start = end

    return partitions


def write_image_with_geometry(img_data: np.ndarray, ref_img: sitk.Image, outfile: str):
    shifted_img = sitk.GetImageFromArray(img_data)

    shifted_img.SetOrigin(ref_img.GetOrigin())
    shifted_img.SetSpacing(ref_img.GetSpacing())
    shifted_img.SetDirection(ref_img.GetDirection())

    print(f"Writing output to: {outfile}...")
    try:
        sitk.WriteImage(shifted_img, outfile)
    except Exception as e:
        raise RuntimeError(f"Could not write image to file: {outfile}") from e

    print('All finished.')
