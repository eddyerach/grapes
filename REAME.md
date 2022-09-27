# Bunch Detector

# Detector and Berry Counting

# Bunch Count (Tracker on detector)

## Table of Content 

- [Installation](#Installation)
- [Data Configuration (images, labels)](#DataConfiguration)
- [Training Guide](#TrainingGuide)
- [Inference Guide](#InferenceGuide)
- [Result](#Result)
- [Metric Table](#MetricTable)

# Installation
First, we need to know that this project was carried out on a computer with the following features:
  - OS type: 64-bit  Ubuntu/Linux 18.04.6 LTS
  - Processor: 11th Gen Intel Core i7-11700 @2.50GHz x 16
  - Graphics: NVIDIA GeForce RTX 3060
  
 ### Docker commands
 Download image 
 ```
sudo docker pull eddyerach1/detectron2_banano_uvas:latest
```
Build a container from an image
```
sudo docker run --gpus all -it -v /home/grapes:/shared  --name detectron2_grapes
```
# Data Configuration (images, labels) 
## For Training Model 
### Dataset for training 
For training the model we use 9 datasets (see the table). Images were labeled with **VGG Imagen Annotator software**  and using the polygon tool. This tool is important because it generate 3 points which allows us to draw through the image contour. 
|Dataset|Images with label |Total labels|
|-------|------------------|------------|
| Dataset 1 (Train & Val)  | 72  |  747 | 
| Dataset 2 (Train & Val)  | 34  |  36  |     
| Dataset 3 (Train & Val)  | 55  |  67  |
| Dataset 4 (Train & Val)  | 46  |  454 |
| Dataset 5 (Train & Val)  | 46  |  585 |
| Dataset 6 (Train & Val)  | 30  |  518 |
| Dataset 7 (Train & Val)  | 31  |  391 |
| Dataset 8 (Train & Val)  | 29  |  454 |
| Dataset 9 (Train & Val)  | 30  |  217 |
# Training Guide

# Inference Guide

# Results

# Metric Table
