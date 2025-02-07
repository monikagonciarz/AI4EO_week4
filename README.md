# AI4EO_week4
GEOL0069 Assignment 4

Sea-ice and lead altimetry classification - unsupervised learning methods

This project is the assessed practical for GEOL0069 week 4. The goal is to use unsupervised learning methods to classify sea ice and leads from Sentinel-3 altimetry datasets, and produce an average echo shape and standard deviation for these two classes. Then, the results are quantified against the ESA official classification using a confusion matrix. The Assignment4_Unsupervised_Learning_Methods.ipynb notebook available in this GitHub is a development of the notebook Chapter1_Unsupervised_Learning_Methods.ipynb provided by the GEOL0069 module team: Dr Michel Tsamados, Weibin Chen, and Connor Nelson.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
- [Results](#results)

---

## Introduction






## Installation



```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
pip install rasterio
```
```python
pip install netCDF4
```

## Gaussian Mixture Models (GMM)


## Results






![1](https://github.com/user-attachments/assets/9ad1fa9f-5d32-4703-b970-946ab34ba70f)




![2](https://github.com/user-attachments/assets/246ef65c-3cb9-48f2-ba9a-bc281ed113cf)



![3](https://github.com/user-attachments/assets/dc1f93c1-6491-4417-8e7f-918aed4db2cd)




![4](https://github.com/user-attachments/assets/15e20798-5d73-493a-862d-a4441acab884)



Below is a confusion matrix quantofying the ESA official classification (flags) against the GMM cluster classification:

![5](https://github.com/user-attachments/assets/7f70d127-1c57-4d6b-be02-2de897e7e24a)


