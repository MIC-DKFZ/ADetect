"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

import SimpleITK as sitk
import numpy as np


def calculate_segmentation_volumes(segmentation_image: sitk.Image) -> dict:
    segmentation_array = sitk.GetArrayFromImage(segmentation_image)

    voxel_spacing = segmentation_image.GetSpacing()
    voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]

    volumes = {}

    unique_labels, label_voxel_count = np.unique(segmentation_array, return_counts=True)
    for label, count in zip(unique_labels, label_voxel_count):
        if label == 0:
            continue

        label_volume_ml = count * voxel_volume * 1e-3

        volumes[str(label)] = {
            "voxel_count": count,
            "volume_ml": label_volume_ml,
        }

    return volumes
