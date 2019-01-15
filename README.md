# Overview

## Weakly supervised localization :
In this task, we have to plot bounding boxes for pneumonia in a single chest X-ray without goundtruth (X, Y, width, height) in training set. The workflow is shown below:

### Workflow :
1) Predict findings
2) Use the classifier to plot heatmap 
3) Plot the bounding box base on heatmap

