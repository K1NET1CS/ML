# About this repository
- Compares performance of CNN based feature extractor (ResNet50 Layer 2,3) and ViT based feature extractor (DINOv2-(ViT-B/14)) on common heads : auto encoder, auto encoder with position encoding, Patch Distribution Modelling
- 


# Anomaly Detection on MVTec AD: CNN and Statistical Methods

This repository contains Jupyter Notebook implementations and evaluations of various Convolutional Neural Network (CNN) and statistical models for unsupervised anomaly detection. The models are evaluated on the MVTec AD dataset, with performance measured using Area Under the Receiver Operating Characteristic (AUROC) for both image-level anomaly detection and pixel-level anomaly localization.

## Performance Metrics

| Category | PerPixelReconstructor (Image) | PerPixelReconstructor (Pixel) | SpatialReconstructor (Image) | SpatialReconstructor (Pixel) | ae (Image) | ae (Pixel) | ae_pos (Image) | ae_pos (Pixel) | ensemble (Image) | ensemble (Pixel) | padim (Image) | padim (Pixel) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bottle | 0.9857 | 0.9679 | 0.8667 | 0.9096 | 1.0000 | 0.9779 | 1.0000 | 0.9778 | 1.0000 | 0.9761 | 1.0000 | 0.9705 |
| Cable | 0.6900 | 0.8983 | 0.7421 | 0.8697 | 0.8632 | 0.9630 | 0.8596 | 0.9639 | 0.9190 | 0.9767 | 0.9505 | 0.9769 |
| Capsule | 0.5166 | 0.9416 | 0.3909 | 0.9071 | 0.7794 | 0.9848 | 0.7886 | 0.9851 | 0.8229 | 0.9856 | 0.8955 | 0.9835 |
| Carpet | 0.8499 | 0.9800 | 0.7576 | 0.9248 | 0.9157 | 0.9843 | 0.9109 | 0.9841 | 0.9506 | 0.9847 | 0.9912 | 0.9836 |
| Grid | 0.5973 | 0.8180 | 0.2807 | 0.7203 | 0.6174 | 0.9228 | 0.6023 | 0.9226 | 0.6951 | 0.9399 | 0.8296 | 0.9496 |
| Hazelnut | 0.9400 | 0.9780 | 0.9646 | 0.9757 | 0.9993 | 0.9831 | 0.9996 | 0.9830 | 0.9829 | 0.9824 | 0.9200 | 0.9786 |
| Leather | 0.9840 | 0.9937 | 0.9460 | 0.9600 | 1.0000 | 0.9871 | 1.0000 | 0.9870 | 1.0000 | 0.9843 | 1.0000 | 0.9784 |
| Metal Nut | 0.6496 | 0.8801 | 0.4648 | 0.8144 | 0.9961 | 0.9721 | 0.9966 | 0.9722 | 0.9971 | 0.9714 | 0.9858 | 0.9591 |
| Pill | 0.7256 | 0.9242 | 0.5060 | 0.9208 | 0.8944 | 0.9864 | 0.8969 | 0.9866 | 0.9176 | 0.9866 | 0.9558 | 0.9771 |
| Screw | 0.5007 | 0.9403 | 0.4237 | 0.9270 | 0.7053 | 0.9690 | 0.7131 | 0.9691 | 0.7565 | 0.9782 | 0.7977 | 0.9751 |
| Tile | 0.7846 | 0.9034 | 0.8810 | 0.8862 | 0.9877 | 0.9452 | 0.9867 | 0.9447 | 0.9986 | 0.9478 | 1.0000 | 0.9456 |
| Toothbrush | 0.6056 | 0.9328 | 0.5917 | 0.6999 | 0.8111 | 0.9843 | 0.7944 | 0.9842 | 0.8444 | 0.9843 | 0.8944 | 0.9815 |
| Transistor | 0.5975 | 0.6846 | 0.6417 | 0.6992 | 0.8167 | 0.8916 | 0.8146 | 0.8995 | 0.9058 | 0.9714 | 0.9858 | 0.9816 |
| Wood | 0.9772 | 0.9393 | 0.9061 | 0.8547 | 0.9842 | 0.9349 | 0.9807 | 0.9344 | 0.9868 | 0.9344 | 0.9825 | 0.9251 |
| Zipper | 0.6762 | 0.9377 | 0.6836 | 0.8086 | 0.9354 | 0.9718 | 0.9367 | 0.9724 | 0.9575 | 0.9752 | 0.9795 | 0.9730 |
| **AVERAGE** | **0.7387** | **0.9147** | **0.6698** | **0.8585** | **0.8871** | **0.9639** | **0.8854** | **0.9644** | **0.9157** | **0.9719** | **0.9446** | **0.9693** |

## Model Summaries

**1. PerPixelReconstructor**
A baseline reconstruction architecture that attempts to recreate the input image exactly. It detects anomalies by calculating the independent, pixel-wise error between the input and the reconstructed output, flagging areas with high deviation as anomalous.

**2. SpatialReconstructor**
A modified reconstruction model that emphasizes the structural and spatial context of the image rather than isolated pixels. By focusing on local neighborhoods and feature patches, it aims to reduce false positives caused by minor pixel shifts and better identify structural defects.

**3. AE (Standard Convolutional Autoencoder)**
An unsupervised learning model that compresses input images into a lower-dimensional latent bottleneck before expanding them back to the original dimensions. Because it is trained solely on normal data, it struggles to accurately reconstruct unseen anomalous features, resulting in a high reconstruction error at defect sites.

**4. AE_Pos (Autoencoder with Positional Encoding)**
An enhancement of the standard autoencoder that integrates spatial or positional encodings into the network. This allows the model to retain a strict awareness of the global layout and geometric locations of features, improving defect localization on structurally rigid objects.

**5. Ensemble**
A robust approach that aggregates the predictions of multiple reconstruction models or autoencoders. By averaging the anomaly scores from varying architectures or training initializations, it reduces model variance, suppresses noise, and yields a more stable and accurate overall detection performance.

**6. PaDiM (Patch Distribution Modeling)**
A powerful statistical anomaly detection method that leverages pre-trained CNNs (such as ResNet) for feature extraction. It avoids a reconstruction bottleneck entirely; instead, it models the normal distribution of localized patch embeddings using a multivariate Gaussian. Anomalies are scored using the Mahalanobis distance between a test patch and the learned normal distribution.


# Anomaly Detection on MVTec AD: ViT extractor followed by same heads (AE,AE_pos,PaDiM)

ViT (DinoV2) Feature Extractor

This feature extractor uses weights from blocks 5 and 8 of the pre-trained Dinov2_vitb14 model. By leveraging Vision Transformers (ViT) instead of traditional CNNs, this method captures deep, global contextual representations of the input images. These rich, dense features are then passed to the anomaly detection heads (AE, AE_Pos, PaDiM, and Ensemble). The ViT backbone often yields superior detection performance by better understanding long-range dependencies within the image.

## ViT-Based Extractor Performance

| Category | ae (Image) | ae (Pixel) | ae_pos (Image) | ae_pos (Pixel) | padim (Image) | padim (Pixel) | ensemble (Image) | ensemble (Pixel) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Bottle | 0.9992 | 0.9895 | 0.9992 | 0.9892 | 1.0000 | 0.9897 | 1.0000 | 0.9904 |
| Cable | 0.9239 | 0.9676 | 0.9278 | 0.9686 | 0.8731 | 0.9792 | 0.8919 | 0.9805 |
| Capsule | 0.9581 | 0.9868 | 0.9581 | 0.9866 | 0.8983 | 0.9897 | 0.9186 | 0.9903 |
| Carpet | 1.0000 | 0.9961 | 0.9996 | 0.9962 | 0.9988 | 0.9960 | 0.9992 | 0.9962 |
| Grid | 1.0000 | 0.9959 | 1.0000 | 0.9959 | 0.9975 | 0.9940 | 0.9992 | 0.9948 |
| Hazelnut | 0.9986 | 0.9947 | 0.9989 | 0.9945 | 0.5621 | 0.9866 | 0.6961 | 0.9915 |
| Leather | 1.0000 | 0.9945 | 1.0000 | 0.9944 | 0.9929 | 0.9945 | 0.9997 | 0.9946 |
| Metal Nut | 1.0000 | 0.9622 | 1.0000 | 0.9621 | 0.9995 | 0.9731 | 1.0000 | 0.9738 |
| Pill | 0.9776 | 0.9684 | 0.9785 | 0.9676 | 0.9351 | 0.9484 | 0.9468 | 0.9559 |
| Screw | 0.8692 | 0.9898 | 0.8717 | 0.9896 | 0.9078 | 0.9937 | 0.9383 | 0.9945 |
| Tile | 1.0000 | 0.9806 | 1.0000 | 0.9809 | 0.9964 | 0.9771 | 0.9978 | 0.9780 |
| Toothbrush | 1.0000 | 0.9943 | 1.0000 | 0.9942 | 0.9917 | 0.9933 | 0.9917 | 0.9933 |
| Transistor | 0.9779 | 0.9036 | 0.9779 | 0.9033 | 0.9550 | 0.9766 | 0.9592 | 0.9758 |
| Wood | 0.9895 | 0.9753 | 0.9904 | 0.9757 | 0.9895 | 0.9700 | 0.9912 | 0.9724 |
| Zipper | 0.9932 | 0.9845 | 0.9921 | 0.9849 | 0.9905 | 0.9847 | 0.9937 | 0.9854 |
| **AVERAGE** | **0.9791** | **0.9789** | **0.9796** | **0.9789** | **0.9392** | **0.9831** | **0.9549** | **0.9845** |

## Backbone Comparison: CNN vs ViT (Image AUROC)

The following table compares the overall Image AUROC performance between the baseline CNN extractor and the ViT extractor.

| Category | CNN | ViT | Delta (ViT - CNN) |
| :--- | :--- | :--- | :--- |
| Bottle | 1.0000 | 0.9996 | -0.0004 |
| Cable | 0.8981 | 0.9042 | 0.0061 |
| Capsule | 0.8216 | 0.9333 | 0.1117 |
| Carpet | 0.9421 | 0.9994 | 0.0573 |
| Grid | 0.6861 | 0.9992 | 0.3131 |
| Hazelnut | 0.9754 | 0.8139 | -0.1615 |
| Leather | 1.0000 | 0.9981 | -0.0019 |
| Metal_nut | 0.9939 | 0.9999 | 0.0060 |
| Pill | 0.9162 | 0.9595 | 0.0433 |
| Screw | 0.7431 | 0.8968 | 0.1536 |
| Tile | 0.9932 | 0.9986 | 0.0053 |
| Toothbrush | 0.8361 | 0.9958 | 0.1597 |
| Transistor | 0.8807 | 0.9675 | 0.0868 |
| Wood | 0.9836 | 0.9901 | 0.0066 |
| Zipper | 0.9523 | 0.9924 | 0.0401 |
| **MEAN** | **0.9082** | **0.9632** | **0.0551** |

