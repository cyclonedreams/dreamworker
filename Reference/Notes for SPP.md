# Notes for *SPP*
## 1 Introduction
A technical issue: the prevalent CNNs require a fixed input image size, which limits both the aspect ratio and the scale of the input image. Cropped region may not contain the entire object, while the warped content may result in unwanted geometric distortion.

**Why do CNNs require a fixed input size?**

A CNN mainly consists of two parts: convolutional layers, and fully-connected layers that follow. In fact, convolutional layers do not require a fixed image size and can generate feature maps of any sizes. On the other hand, the fully-connected layers need to have fixed-size/length input by their definition. Hence, the fixed-size constraint comes only from the fully-connected layers, which exist at a deeper stage of network.

In this paper, we introduce a spatial pyramid pooling (SPP) layer to remove the fixed-size constraint of the network. Specifically, we add an SPP layer on top of the last convolutional layer. The SPP layer pools the features and generates fixed-length outpus, which are then fed into the fully-connected layers (or other classifiers). In other words, we perform some "aggregation" at a deeper stage of the network hierarchy to avoid the need for cropping or warping at the beginning.

**Conv network + SPP layer - cropping/warping = SPP-net**

## 2 Deep Networks with Spatial Pyramid Pooling
### 2.1 Convolutional Layers and Feature Maps
Conv layers -- Feature Maps
### 2.2 The Spatial Pyramid Pooling Layer
The classifiers (SVM/softmax) or fully-connected layers require fixed-length vectors.
To adopt the deep network for images of arbitrary sizes, we replace the last pooling layer (e.g., pool5, after the last convolutional layer) with SPP layer.