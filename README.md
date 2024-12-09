<img src=docs/source/automatic_results.png width="700px">

# ADetect
This project is aimed at creating a fast and reliable tool for detecting aortic dissections (AD) in CT scans tailored for emergency scenarios. To achieve this, we trained and optimized deep segmentation networks using data collected at an academic-level hospital in Germany, over a span of more than 10 years. Data preparation and annotation processes were designed to support the development and validation of the presented method. The self-configuring [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/master) framework was utilized to optimize segmentation performance. A robust detection of AD is achieved by applying a refined thresholding rule to the segmentation output of the trained networks, effectively integrating over the segmented AD-specific structures. 

## Key Features
- **Robust and accurate detection**, with performance metrics reflecting AUC greater than 0.97, sensitivity above 92%, and specificity close to 100%, validated across multiple datasets from diverse sites.
- **Fully-automated and streamlined**, with processing time of less than 7 minutes per case, enabling seamless integration into clinical workflows.
- **Proven effectiveness** in identifying AD cases that do not exhibit typical symptoms and are often under-prioritized in emergency scenarios.

## Current Status
Our evaluation results and key findings are undergoing peer review and will be published soon. Once finalized, the trained models will be made available here for further scientific validation and collaborative research. 

Stay tuned.

## Important Notes
- The proposed tool has been trained and validated exclusively on de-identified data in a research setting and has not undergone clinical testing.
- It is not a medical product and is intended for research purposes only.
- No patient data or personal information have been published in this project.
- The automatic segmentations included in this repository and used for integration testing are based exclusively on a publicly available dataset.
