# Overview

## Weakly supervised localization :
In this task, we have to plot bounding boxes for each disease finding in a single chest X-ray without goundtruth (X, Y, width, height) in training set. The workflow is shown below:
### Workflow :
1) Predict findings
2) Use the classifier to plot heatmap (Grad-CAM)
3) Plot the bounding box base on Grad-CAM
