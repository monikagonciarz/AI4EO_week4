# AI4EO_week4
GEOL0069 Assignment 4

Sea-ice and lead altimetry classification - unsupervised learning methods

This project is the assessed practical for GEOL0069 week 4. The goal is to use unsupervised learning methods to classify sea ice and leads from Sentinel-3 altimetry datasets, and produce an average echo shape and standard deviation for these two classes. Then, the results are quantified against the ESA official classification using a confusion matrix. The Assignment4_Unsupervised_Learning_Methods.ipynb notebook available in this GitHub is a development of the notebook Chapter1_Unsupervised_Learning_Methods.ipynb provided by the GEOL0069 module team: Dr Michel Tsamados, Weibin Chen, and Connor Nelson.

<br>  

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

<br>  

## Introduction

This project was developed using Google Colab, a cloud-based environment that makes it easy to write, run, and share Python code collaboratively. Colab offers free access to high-performance GPU and TPU resources, without the need for expensive hardware. It integrates with Google Drive for effortless notebook storage and sharing, as well as collaboration. This makes it an excellent platform for data science, machine learning, and development projects. You can access the notebook by downloading it and uploading it onto your Google Drive, and subsequently opening it with Google Colab for easy code running.


The Sentinel-3 mission primarily focuses on capturing detailed measurements of the sea surface topography. This information is crucial for understanding variations in sea level, the behavior of sea ice, wind speeds over the ocean, and other ocean dynamics such as currents, waves, eddies, and tides. For additional details on this mission, you can visit the Copernicus Sentinel Online Portal.


Altimetry satellites work by transmitting radar signals towards the Earth. When these signals bounce back, the returned portion—known as the echo—provides insights into surface elevations based on the time it takes to return. Differences in the physical properties of various surface materials affect the echo's shape and intensity, enabling us to differentiate between features like sea ice and leads, which is a key aspect of this project.





<br>  

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



<br>  

## Gaussian Mixture Models (GMM)


This project employs Gaussian Mixture Models (GMMs) as its unsupervised learning method. GMMs are probabilistic models used to represent normally distributed subpopulations within an overall population. They assume that the data are generated from a mixture of several Gaussian distributions, each with its own mean and variance (and thus also its own standard deviation). This approach enables complex distributions to be modeled as combinations of simpler Gaussian distributions, making GMMs especially useful for clustering and density estimation.

The key components of a GMM include:

- The number of components (Gaussians): Specifies how many distinct subpopulations are assumed.
- The expectation-maximization algorithm: An iterative process that computes the probability that each data point belongs to each cluster and then updates the model parameters to maximize the likelihood of the observed data.
- The covariance type: Determines the shape and orientation of the Gaussian components.
Other unsupervised learning methods include K-means clustering. (While support vector machines (SVMs) are primarily used for supervised learning, unsupervised variants like one-class SVM can be applied for anomaly detection.)

The Gaussian model can be initialized using the GaussianMixture function from sklearn.mixture. When initializing, you need to define the number of components (or clusters) and the random state, which helps ensure that the results are reproducible. After setting up the model, it can be fitted to the data using `gmm.fit()`, based on the way the data is preprocessed or cleaned. Finally, the model can predict the cluster labels for the data points using `gmm.predict()`, the size of which can be checked using `clusters_gmm.shape`. The size of them should match `waves_cleaned.shape`, `data_cleaned.shape`, and `flag_cleaned.shape`.

We can also inspect how many data points there are in each class of the clustering predictions.
```python
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Cluster counts:", class_counts)

# From the outputs: 8880 pixels classified as sea ice, 3315 pixels classified as leads
```

<br>  




## Results

The first step in this project was to classify the echoes in leads and sea ice. Function clusters made with the help of the Gaussian Mixture Model are extracted and plotted using plt from matplotlib.pyplot. The sea ice cluster echoes are labeled '0' and the lead cluster echos are labeled '1' This means that `waves_cleaned[clusters_gmm = 0`] are sea ice echoes, and `waves_cleaned[clusters_gmm = 1`] indicate lead echoes. The plot below shows classified clusters for 10 equally spaced functions of each class, sea ice and leads, for simplicity.

![1](https://github.com/user-attachments/assets/9ad1fa9f-5d32-4703-b970-946ab34ba70f)

<br>

It can be seen that the lead waveforms are very large, spikey pulses (most close to 10000), while the sea ice peaks are much lower and broader (most around 1000). The aligned plots on the right have been adjusted by shifting the waveforms within each cluster so that their echo peaks are aligned, achieved through cross-correlation.


<br>
<br>
<br>

The next step was to produce an average echo shape and standard deviation for the two analysed classes. The plots of the means and standard deviations for the 10 equally spaced functions, and also for all functions, are presented below. 

![2](https://github.com/user-attachments/assets/246ef65c-3cb9-48f2-ba9a-bc281ed113cf)

<br>


![3](https://github.com/user-attachments/assets/dc1f93c1-6491-4417-8e7f-918aed4db2cd)


<br>



The results indicate that the power of sea ice echoes is much lower than that of leads. Additionally, the lead echoes also exhibit more noise compared to sea ice. These differences might be explained by variations in surface roughness (possibly due to wind or melting and freezing processes), as smoother surfaces tend to be more reflective. 

The standard deviations for sea ice and lead echoes are also calculated and plotted. The results reveal greater variability in the shape of lead echoes compared to sea ice, confirming the earlier observation that lead echoes exhibit more noise.



<br>
<br>
<br>

In the plot below, the means and standard deviations of sea ice and leads, unaligned and aligned, are presented separately. The statistical parameters are plotted for aligned echoes because some of the observed noise in both plots could be due to echo shifts along the x-axis. Aligning the functions addresses this and creates smoother plots with less visible noise.

![4](https://github.com/user-attachments/assets/15e20798-5d73-493a-862d-a4441acab884)

<br>
<br>
<br>

Below is the confusion matrix quantifying the ESA official classification (flags) against the GMM cluster classification.

![5](https://github.com/user-attachments/assets/7f70d127-1c57-4d6b-be02-2de897e7e24a)

<br>  

The confusion matrix offers a way to evaluate the clustering performance:


- Top-left corner (8856): True Positives (TP) — Correct classification of sea ice by both ESA and GMM.
- Top-right corner (22): False Negatives (FN) — Misclassification of sea ice as leads by GMM.
- Bottom-left corner (24): False Positives (FP) — Misclassification of leads as sea ice by GMM.
- Bottom-right corner(3293): True Negatives (TN) — Correct classification of leads by both ESA and GMM.


The matrix suggests a high level of agreement between the ESA official classification and the GMM clustering. The low FN (22) and FP (24) values indicate that GMM rarely makes errors in distinguishing between sea ice and leads, with a strong accuracy overall.

A classification report is presented at the end of the code notebook. It provides metrics such as precision (how many predicted positives are correct), recall (how many actual positives are identified), F1-score (the balance between precision and recall), and support (the number of actual instances for each class in the dataset), which further help evaluate the model's performance.

The outputs are provided here (class 0.0 -> sea ice, class 1.0 -> leads):
          
```python
Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195
```



<br>  

## Acknowledgements
The information used in this file is sourced from the GEOL0069: Artificial Intelligence For Earth Observation (AI4EO) 24/25 UCL Moodle page, created by the module team: Dr Michel Tsamados, Weibin Chen, and Connor Nelson.
