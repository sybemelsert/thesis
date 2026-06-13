# Thesis Title

This repository contains the files I created to perform my experiment on the performance of a segmentation model, such as Segment Anything Model 3 (SAM 3) by Meta AI, on videos with decreased interpretabilities.

## Downloads
The model used in this experiment is *sam3.pt*. This model can be found on huggingface.co, through the [Official Ultralytics Webpage](https://www.example.com).


### *video_converter.py*
This file allows us to easily create modified copies of standard .avi files to base our experiment on. For each *.avi* file it finds, it will look for a *.txt* that matches in name. Once verified, it starts reproducing the video with a slight adjustment for each of the following types:

- Gaussian Blur
- Motion Blur
- Overexposure
- Underexposure
- Occlusion

To accurately review effects, the following three tiers of adjustment have been created: mild, moderate and severe.

### *sam3.py*
