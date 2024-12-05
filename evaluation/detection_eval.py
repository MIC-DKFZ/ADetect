"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

from evaluation.classifier_metrics import get_classifier_metrics, get_roc
import pandas as pd
import numpy as np


def perform_evaluation(seg_data: pd.DataFrame):
    ascending = seg_data['false_lumen_ascending']
    descending = seg_data['false_lumen_descending']
    membrane = seg_data['membrane']
    gt_labels = seg_data['is_AD']
    result = {}

    dataset_description = {
        'Number of positives': sum(gt_labels),
        'Number of negatives': len(gt_labels) - sum(gt_labels),
        'GT vector': gt_labels
    }
    result['Dataset description'] = dataset_description

    decision_values = pd.Series(np.median([ascending, descending, membrane], axis=0), index=ascending.index)
    roc, thresholds = get_roc(gt_labels, decision_values)
    result['ROC analysis'] = roc

    best_thresholds = thresholds[roc['youden_index']:-1]
    detection_performance = {}
    for i, decision_threshold in enumerate(best_thresholds):
        labels_pred = decision_values >= decision_threshold
        classifier_metrics_current = get_classifier_metrics(gt_labels, labels_pred)
        performance = {
            'Decision threshold': decision_threshold,
            'Performance:': classifier_metrics_current
        }
        detection_performance[f"Youden + {i}"] = performance

        if 'false_lumen_ascending_gt' in seg_data.columns:
            ascending_true_positive = ascending[labels_pred & gt_labels]
            true_positive_sfa_cases = seg_data['false_lumen_ascending_gt'][labels_pred & gt_labels] > 0
            roc_stanford_current, thresholds_current = \
                get_roc(true_positive_sfa_cases, ascending_true_positive)

            threshold_ascending = thresholds_current[roc_stanford_current['youden_index']]
            key = f"Stanford classification"
            binary_predictions = (ascending_true_positive >= threshold_ascending).astype(int)
            current_metrics = get_classifier_metrics(true_positive_sfa_cases, binary_predictions)
            performance = {
                'Decision threshold': threshold_ascending,
                'Performance:': current_metrics
            }
            detection_performance[f"Youden + {i}"][key] = performance

    result['Detection performance'] = detection_performance

    return result
