"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, \
    f1_score, auc, roc_curve


def get_roc(labels_gt, decision_values):
    fpr, tpr, thresholds = roc_curve(labels_gt, decision_values)
    roc_auc = auc(fpr, tpr)
    youden_index = tpr - fpr
    best_threshold_idx = int(np.argmax(youden_index))

    results = {'fpr': fpr,
               'tpr': tpr,
               'roc_auc': roc_auc,
               'youden_index': best_threshold_idx,
               'decision_values': decision_values}

    return results, thresholds


def get_classifier_metrics(labels_gt, labels_pred):
    tn, fp, fn, tp = confusion_matrix(labels_gt, labels_pred).ravel()
    sensitivity = recall_score(labels_gt, labels_pred)
    specificity = tn / (tn + fp)
    precision = precision_score(labels_gt, labels_pred)
    npv = tn / (tn + fn)
    accuracy = accuracy_score(labels_gt, labels_pred)
    f1 = f1_score(labels_gt, labels_pred)

    results = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'sensitivity': sensitivity,
               'specificity': specificity, 'precision': precision, 'npv': npv,
               'accuracy': accuracy, 'f1': f1, 'predicted': labels_pred}

    return results
