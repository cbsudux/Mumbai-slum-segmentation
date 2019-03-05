# Training Details.

This file contains details about training and inference for slum segmentation using Mask RCNN.



## Installation
Use this Google Drive link to download the weights: 
* Download `mask_rcnn_slum_600_00128.h5` and save it in root directory.
* Link : https://drive.google.com/file/d/1IIMZLrdCZXY_dA540Ve9lSJplYHLnTY4/view?usp=sharing

## Dataset
Dataset of satellite images can be created using Google Earth's desktop application. For our project, we used 720X1280 images at 1000m and 100m views, from various Mumbai slums. Google's policy states that we cannot redistribute the dataset. 

Also, we recommend using VGG Image Annotator tool for annotating the segmentation masks as the code is written for that format. The tool gives the annotations in the form of a JSON file, that should be placed inside the dataset folder as follows:
```
dataset/
        train/
            all training images
            train.json
        val/
            all val images
            val.json
```

Here are few links to help you to curate your own dataset: <br>
https://productforums.google.com/forum/#!msg/maps/8KjNgwbBzwc/4kNMfXB6CAAJ <br>
https://support.google.com/earth/answer/148146?hl=en <br>
http://www.robots.ox.ac.uk/~vgg/software/via/ <br>
## Training the model

Train a new model starting from pre-trained COCO weights
```
python3 slum.py train --dataset=/path/to/slum/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 slum.py train --dataset=/path/to/slum/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 slum.py train --dataset=/path/to/slum/dataset --weights=imagenet
```

* The training details are specified inside slum.py code.
* The model will save every checkpoint in root/logs folder.
* The logs folders are timestamped according to start time and also have tensorboard visualizations.


## Inference
Testing mode, where a segmentation mask is applied on the detected instances. Make sure to place the images inside ```test_images``` folder.

```bash
python3 testing.py  --weights=/path/to/mask_rcnn/mask_rcnn_slum.h5 
```
This will save the detections (if any) for all the images in `test_images` and save it in `test_outputs`.

Apply splash effect on a video. Requires OpenCV 3.2+:
Segments out instances and applies masks on a video.
```bash
python3 slum.py splash --weights=/path/to/mask_rcnn/mask_rcnn_slum.h5 --video=<file name or URL>
```
## Change Detection
For detecting percentage change in masks, place the two images in ```change_det/ ``` folder and run:

```bash
python3 change_detection.py  --weights=/path/to/mask_rcnn/mask_rcnn_slum.h5 
```
