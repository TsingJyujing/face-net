# Face Net

## Introduction

An implementation of these papers:

- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
- [
FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

The aim is to detect the face in images by the P-Net and O-Net in [Joint Face Detection](https://arxiv.org/abs/1604.02878) and extract the feature vector by [FaceNet](https://arxiv.org/abs/1503.03832) (Maybe we can call it E-Net which 'E' means Embedding).


## Data Set

[WIDER FACE: A Face Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) will be used for face detection training and [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) will be used as face embedding training.
