# Skeleton-Based Human Activity Discovery Technique Using Particle Swarm Optimization with Gaussian Mutation
```BibTeX
@misc{duan2021revisiting,
      title={Revisiting Skeleton-based Action Recognition},
      author={Haodong Duan and Yue Zhao and Kai Chen and Dian Shao and Dahua Lin and Bo Dai},
      year={2021},
      eprint={2104.13586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Introduction

This repository contains the implementation of our proposed method for [human activity discovery](). Human activity discovery aims to distinguish the activities performed by humans, without any prior information of what defines each activities. 
![arch](/Figures/fig-1.jpg)

The workflow of the proposed method is as follows. First, keyframes are selected from the video sequence by computing kinetic energy. Then, features based on different aspects of skeleton including displacement, orientation, and statistical are extracted. Principal components are then chosen by applying PCA on the features. Next, overlapping time windows is used to segment a series of keyframes as activity samples. Hybrid PSO clustering with Gaussian mutation operator is used to discover the groups of activities. Eventually, K-means clustering is applied to the resultant cluster centers to refine the centroids.
![arch](/Figures/fig-2.jpg)




### Results
* 3D Action Classification Results
![arch](images/results_classification.png)

* 3D Action  Retrieval Results
![arch](images/results_retrieval.png)

