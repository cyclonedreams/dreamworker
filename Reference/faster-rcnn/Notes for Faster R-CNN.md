# Notes for *Faster R-CNN*
## 1 Introduction
Proposals are the test-time computational bottelneck in state-of-the-art detection systems.

Selective Search, one of the most popular methods, greedily merges superpixels based on engineered low-level features, but is an order of magnitude slower. EdgeBoxes currently provides the best tradeoff between proposal quality and speed, at 0.2 seconds per image, but still consumes as much running time as the detection network.

Replementing region proposal methods on GPUs may cause misssing important oppotunities for sharing computation.

In this paper, we show that an algorithmic change -- computing proposals with a deep convolutional neural network -- leads to an elegant and effective solutoin where proposal computation is nearly cost-free given the detection network's computation. To this end, we introduce novel Region Proposal Networks (RPNs) that share convolutional layers with state-of-the-art object detection networks. By sharing convolutions at test-time, the marginal cost for computing proposals is small.

Our observation is that the convolutional feature maps used by region-based detectors, like Fast R-CNN, can also be used for generating region proposals. On top of these convolutional features, we construct an RPN by adding a few additional convolutional layers that simultaneously regress region bounds and objectness scores at each location on a regular grid. The RPN is thus a kind of fully convolutional network (FCN) and can be trained end-to-end specifically for the task for generating detection proposals.

RPNs are designed to efficiently predict region proposals with a wide range of scales and aspect ratios. In contrast to prevalent methods that use pyramids of images or pyramids of filters, we introduce novel "anchor" boxes that serve as references at multiple scales and aspect ratios. Our scheme can be thought of as a pyramid of regression references, which avoids enumerating images or filters of multiple scales or aspect ratios.
## 2 Related Work
### Object Proposals.
There is a large literature on object proposals methods. Widely used object proposal methods include those based on grouping super-pixels (e.g., Selective Search, CPMC, MCG) and those based on sliding windows (e.g., objectness in windows, EdgeBoxes).
### Deep Networks for Object Detection.
The R-CNN method trains CNNs end-to-end to classify the proposal regions into object categories or background. It mainly plays as a classifier, and it does not predict object bounds (except for refining by bounding box regression).

Shared computation of convolutions has been attracting increasing attention for efficient, yet accurate, visul recognition. Fast R-CNN enables end-to-end detector training on shared convolutional features and show compelling accuracy and speed.