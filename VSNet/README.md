# Semantic Segmentation pytorch unet

## Description

Implementation Unet convolutional neural network in pytorch for semantic segmentation of CT scan images of chest to detect anomalies.

## Dataset
Contents images in tiff format and their respective mask representing postion of anomalities in original images
Mask image is used to train deep learning model to detect the ROI from whole images

Structure of data folder

data

    --train

        --images
    
        --masks
    
    --test
    
        --images

Sample Image
<div style="text-align: right">
<img src="screenshots/image.jpg" width="200" height="200" />
</div>

Sample Image's mask
<div style="text-align: right">
<img src="screenshots/mask.jpg" width="200" height="200" />
</div>
