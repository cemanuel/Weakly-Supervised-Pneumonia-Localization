# Overview


### Team : Emanuel Cortes (ecortes@stanford.edu), Shih-Cheng (Mars) Huang (mschuang@stanford.edu),  Medi Monam (mmonam@stanford.edu)

### Weakly supervised localization :
In this task, we have to plot bounding boxes for pneumonia in a single chest X-ray without goundtruth (X, Y, width, height) in training set. The workflow is shown below:

### Workflow :
![Alt Text](https://github.com/cemanuel/Weakly-Supervised-Pneumonia-Localization/blob/master/model_architecture.png)
1) Train a binary classifier to predict whether a chest X-ray contains pneumonia.
2) Use the classifier to plot heatmap
3) Applied a depth first search on a random non-zero pixel on the heatmap, and repeat until all non-zero pixels are clustered.
4) Plot the bounding box on each cluster.

### Results:
<img src="https://github.com/cemanuel/Weakly-Supervised-Pneumonia-Localization/blob/master/classification_accuracies.png" width="300" height="150">
<img src="https://github.com/cemanuel/Weakly-Supervised-Pneumonia-Localization/blob/master/iou_scores.png" width="300" height="150">

Sample Localization Predictions (Blue Boundaries are Predictions and Red Boundaries are Ground Truth:
![Alt Text](https://github.com/cemanuel/Weakly-Supervised-Pneumonia-Localization/blob/master/predictions.png)


### Discussion
1.) 
2.) 
3.) 





