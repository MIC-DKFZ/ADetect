"""
SPDX-FileCopyrightText: Copyright 2024 Division of Medical Image Computing,
German Cancer Research Center (DKFZ), Heidelberg, Germany, and contributors

SPDX-License-Identifier: Apache-2.0
"""

import os
import unittest
from unittest.mock import patch
from run import evaluate_detection, prepare_evaluation_data


class IntegrationTestDetectionEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        data_base = "../data/reference_data"
        outbase = "../examples"

        self.volume_data = f"{data_base}/volumes.csv"
        self.outfile = f"{outbase}/results.json"
        self.outfile_sf = f"{outbase}/results_sfa.json"
        self.volume_data_gt = f"{data_base}/volumes_gt.csv"
        self.ref_results_file = f"{data_base}/results.json"
        self.ref_results_file_sf = f"{data_base}/results_sfa.json"
        self.argv = ['script_name',
                     '--segmentation-csv', self.volume_data,
                     '--evaluation-output', self.outfile]

        self.argv_sfa = ['script_name',
                         '--segmentation-csv', self.volume_data,
                         '--evaluation-output', self.outfile_sf,
                         '--ground-truth-csv', self.volume_data_gt]


    def tearDown(self) -> None:
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        if os.path.exists(self.outfile_sf):
            os.remove(self.outfile_sf)


    @patch('sys.argv', new_callable=list)
    def test_evaluate_detection(self, mock_argv):
        mock_argv[:] = self.argv

        evaluate_detection.main()

        with open(self.ref_results_file) as rf:
            ref_performance = rf.readlines()
        with open(self.outfile) as of:
            output_performance = of.readlines()

        self.assertEqual(ref_performance, output_performance)

    @patch('sys.argv', new_callable=list)
    def test_evaluate_detection_stanford(self, mock_argv):
        mock_argv[:] = self.argv_sfa

        evaluate_detection.main()

        with open(self.ref_results_file_sf) as rf:
            ref_performance = rf.readlines()
        with open(self.outfile_sf) as of:
            output_performance = of.readlines()

        self.assertEqual(ref_performance, output_performance)


class IntegrationTestPrepareEvaluationData(unittest.TestCase):

    def setUp(self) -> None:
        data_base = "../data/reference_data"
        outbase = "../examples"

        self.segfolder_diseased = f"{data_base}/seg_niftis/diseased"
        self.segfolder_healthy = f"{data_base}/seg_niftis/healthy"
        self.labelfile = f"{data_base}/labelfile.json"
        self.outfile_volumetry = f"{outbase}/volumes_test.csv"
        self.ref_volumetry = f"{data_base}/volumes_test.csv"
        self.num_processes = 4

        self.argv_prep_data = ['script_name',
                               '--segfolder-diseased', self.segfolder_diseased,
                               '--segfolder-healthy', self.segfolder_healthy,
                               '--labelfile', self.labelfile,
                               '--outfile', self.outfile_volumetry,
                               '--num_processes', str(self.num_processes)]


    def tearDown(self) -> None:
        if os.path.exists(self.outfile_volumetry):
            os.remove(self.outfile_volumetry)


    @patch('sys.argv', new_callable=list)
    def test_prepare_evaluation_data(self, mock_argv):
        mock_argv[:] = self.argv_prep_data

        prepare_evaluation_data.main()

        with open(self.ref_volumetry) as rf:
            ref_volumes = rf.readlines()
        with open(self.outfile_volumetry) as of:
            output_volumes = of.readlines()

        self.assertEqual(ref_volumes, output_volumes)


if __name__ == '__main__':
    unittest.main()
