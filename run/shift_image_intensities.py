"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

import argparse
from utilities.data_io import load_segmentation, write_image_with_geometry
import os
import SimpleITK as sitk


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile',
                        type=str,
                        required=True,
                        help='Input nifti file.')
    parser.add_argument('--outfile',
                        type=str,
                        required=True,
                        help='Output nifti file.')
    parser.add_argument('--intensity_shift', dest='value',
                        type=int,
                        default=-1000,
                        help='Shift image intensities by this amount.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    infile = args.infile
    outfile = args.outfile
    val = args.value

    if not os.path.isfile(infile):
        raise ValueError(f"Input file could not be read: {infile}")

    print(f"Loading file: {infile}...")
    try:
        img = load_segmentation(infile)
    except Exception as e:
        raise RuntimeError(f"Input file: {infile} seems to be no valid nifti file.") from e

    img_data = sitk.GetArrayFromImage(img)
    print(f"Shifting intensities by {val}")
    shifted_img_data = img_data + val

    write_image_with_geometry(shifted_img_data, img, outfile)

    print('All finished.')


if __name__ == '__main__':
    main()
