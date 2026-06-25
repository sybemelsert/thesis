# FOLDER NAME (PROB THESIS NAME)

This repository contains the files I created to perform my experiment on the performance of a segmentation model, such as Segment Anything Model 3 (SAM 3) by Meta AI, on videos with decreased interpretabilities.

## Downloads
The model used in this experiment is *sam3.pt*. This model can be found on huggingface.co, through the [Official Ultralytics Webpage](https://www.example.com).


## Folders

### *DR(eye)VE*
Includes original and modified data samples from the [DR(eye)VE](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=8) dataset, developed by researchers at the University of Modena and Reggio Emilia.

### *images*
Serves as the target folder for statistical output images created in [*visualizer.py*](visualizer.py), such as graphs and tables.

### *results*
Text


## Files REORDER CORRECTLY

### *sam3.pt*
Text

### *video_converter.py*
This file allows us to easily create modified copies of standard .avi files to base our experiment on. For each *.avi* file it finds, it will look for a *.txt* that matches in name. Once verified, it starts reproducing the video with a slight adjustment for each of the following types:

- Overexposure
- Underexposure
- Motion Blur
- Gaussian Blur
- Grain
- Occlusion

To accurately review effects, the following three tiers of adjustment have been created: 
- Mild
- Medium
- Severe

### *sam3.py*
The *sam3.py*-file serves as the core integration between raw data and the [SAM 3](https://ai.meta.com/research/sam3/) model designed by Meta AI.

Several evaluation metrics and control variables are calculated within this file to produce a *.csv*-file., with which conclusions can be drawn on the performance of SAM 3 on decremented video quality.

#### Evaluation Metrics




#### Control Variables

### *visualizer.py*
Text

### *analysis.py*
This file takes a *.csv*-file  

