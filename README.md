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

First, the Google Drive must be mounted on Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```



Then, the following software needs to be installed in order to run the code:

```python
pip install rasterio
```
```python
pip install netCDF4
```

Next, a set of functions must be loaded. It's crucial to preprocess the data to ensure compatibility with our analytical models. This involves transforming the raw data into meaningful variables, such as peakiness and stack standard deviation (SSD), and removing NaN values. At any point, shapes of arrays can be checked using e.g. `waves.shape`. The exact process is detailed in the notebook file.



## Gaussian Mixture Models (GMM)

The Gaussian model can be initialized using the GaussianMixture function from sklearn.mixture. When initializing, you need to define the number of components (or clusters) and the random state, which helps ensure that the results are reproducible. After setting up the model, it can be fitted to the data using `gmm.fit()`, based on the way the data is preprocessed or cleaned. Finally, the model can predict the cluster labels for the data points using `gmm.predict()`, the size of which can be checked using `clusters_gmm.shape`. The sieze of them should match `waves_cleaned.shape`, `data_cleaned.shape`, and `flag_cleaned.shape`.

We can also inspect how many data points are there in each class of your clustering prediction.
```python
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Cluster counts:", class_counts)

# From the outputs: 8880 pixels classified as sea ice, 3315 pixels classified as leads
```


## Results






![1](https://github.com/user-attachments/assets/9ad1fa9f-5d32-4703-b970-946ab34ba70f)




![2](https://github.com/user-attachments/assets/246ef65c-3cb9-48f2-ba9a-bc281ed113cf)



![3](https://github.com/user-attachments/assets/dc1f93c1-6491-4417-8e7f-918aed4db2cd)




![4](https://github.com/user-attachments/assets/15e20798-5d73-493a-862d-a4441acab884)



Below is a confusion matrix quantofying the ESA official classification (flags) against the GMM cluster classification:

![5](https://github.com/user-attachments/assets/7f70d127-1c57-4d6b-be02-2de897e7e24a)


