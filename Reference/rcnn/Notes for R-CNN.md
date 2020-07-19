# Note for *R-CNN*
## 2. Object detection with R-CNN
Our object detection system consists of three modules:

1. generates category-independent region proposals. These proposals define the set of the candidate detections available to our detector.
2. a large convolutional neural network that extracts a fixed-length feature vector from each region.
3. a set of class-specific linear SVMs.
### 2.1. Module design
#### Region proposals.
We use selective search to enable a controlled comparison with prior detection work.
#### Feature extraction.
We extract a 4096-dimensional feature vector from each region proposal using the Caffe implementation of the CNN described by Krizhevsky et al.

In order to compute features for a region proposal, we must first convert the image data in that region into a form that is compatible with the CNN. Of the many possible transformations of our arbitrary-shaped regions, we opt for the simplest. Regardless of the size or aspect ratio of the candidate region, we warp all pixels in a tight bounding box around it to the required size. Prior to warping, we dilate the tight bounding box so that at the warped size there are exactly p pixels of warped image context around the original box (we use p=16)
### 2.2. Test-time detection
1. run selective search on the test image to extract around 2000 region proposals.
2. warp each proposal and forward propagate it through the CNN in order to compute features.
3. for each class, we score each extracted feature vector using the SVM trained for that class.
4. apply a greedy non-maximum suppression for each class independently, reject a region if it has an intersection-over-union (IoU) overlap with a higher scoring selected region larger than a learned threshold.
#### Run-time analysis.