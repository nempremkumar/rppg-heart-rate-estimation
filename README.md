# rppg-heart-rate-estimation
rppg-heart-rate-extraction-
This repository contains the project work on non-invasive physiological monitoring using thermal and multispectral NIR imaging techniques. The primary goal is to extract vital signs, such as heart rate, by processing the imaging data.

Table of Contents

Certainly! Here's a template for your GitHub README based on the details you've provided:

Remote Physiological Monitoring Using Thermal and Multispectral NIR Imaging

This repository contains the project work on non-invasive physiological monitoring using thermal and multispectral NIR imaging techniques. The primary goal is to extract vital signs, such as heart rate, by processing the imaging data.

Table of Contents

Datasets Data Processing Thermal Work Multispectral NIR Work (16 Bandwidths) Installation and Setup Usage License Datasets

Thermal Imaging Dataset: This dataset consists of thermal images captured under various conditions to understand the correlation with physiological phenomena. Multispectral NIR Imaging Dataset: Comprising of 16 different bandwidths, this dataset captures a wide spectrum of information for comprehensive analysis. Data Processing

The data processing phase is crucial to filter out noise and extract meaningful signals from the imaging data. The following methods are employed: Data Processing: Video Synchronization and Visualization One of the key challenges in remote physiological monitoring is ensuring that the data is visualized in a manner that's both intuitive and aligned with real-time events. To address this, our project incorporates a dedicated window for video playback, enabling users to gain a visual representation of the imaging data.

Pipeline for the physiological parameters estimation image
![image](https://github.com/nempremkumar/rppg-heart-rate-estimation/assets/145294904/092d449b-26ba-4a44-b4ac-01226d89bb9b)


Thermal Work: The thermal dataset undergoes various processing stages, including: ROI extraction Temporal signal extraction heart rate extraction

Multispectral Work: The multispectral data, with its 16 distinct bandwidths, offers a rich set of information: Extraction of temporal signals for each bandwidth Smoothing using a window size of 3 Heart rate extraction

Installation and Setup

Usage
