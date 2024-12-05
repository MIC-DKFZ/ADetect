"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

from utilities.data_io import write_results, check_valid_input, read_segmentation_input, merge_seg_data_and_gt
from evaluation.detection_eval import perform_evaluation
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation-csv', '-seg_csv', type=str, required=True,
                        help='Path to a CSV file containing containing segmentation volumes (in milliliters) for AD, '
                             'along with image-level ground-truth labels (AD or non-AD).')
    parser.add_argument('--evaluation-output', '-eval_output', type=str, required=True,
                        help='Path to a json output file where the evaluation results for AD detection will be saved.')
    parser.add_argument('--ground-truth-csv', '-seg_csv_gt', type=str,
                        help='Path to a csv file containing ground-truth segmentation volumes (in milliliters) '
                             'and image-level labels for AD- and non-AD cases.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    seg_csv = args.segmentation_csv
    eval_output = args.evaluation_output
    seg_csv_gt = args.ground_truth_csv

    seg_data = read_segmentation_input(seg_csv)
    try:
        check_valid_input(seg_data)
    except ValueError as e:
        raise ValueError(f"Something is wrong with the segmentation input from file {seg_csv}. "
                         f"Input must have entries for both classes (AD and non-AD) and must not contain NANs.") \
            from e

    if seg_csv_gt is not None:
        seg_data_gt = read_segmentation_input(seg_csv_gt)
        try:
            check_valid_input(seg_data_gt)
        except ValueError as e:
            raise ValueError(f"Something is wrong with the ground truth segmentation input from file {seg_csv_gt}. "
                             f"Input must have entries for both classes (AD and non-AD) and must not contain NANs. ") \
                from e

        try:
            merge_seg_data_and_gt(seg_data, seg_data_gt)
        except ValueError as e:
            raise ValueError(f"Could not merge segmentation data input from file {seg_csv}, "
                             f"with ground truth data from file {seg_csv_gt}") from e

    result = perform_evaluation(seg_data)

    write_results(result, eval_output)

    print("All finished.")


if __name__ == '__main__':
    main()
