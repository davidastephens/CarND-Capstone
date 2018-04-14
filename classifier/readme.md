# Traffic Light Dataset

I created this traffic light dataset from the Autti dataset located here:
https://github.com/udacity/self-driving-car/tree/master/annotations

Inspired by:
https://github.com/datitran/raccoon_dataset/blob/master/README.md

## Getting Started

##### Folder Structure: ```
+ data: contains the input file for the TF object detection API and the label
  files (csv)
+ images: contains the image data in jpg format
+ training: contains the pipeline configuration file, frozen model and labelmap
- generate_tfrecord.py is used to generate the input files for the TF API
- filter_traffic.py filters out the traffic lights from the whole dataset and
  creates a csv that can be used by generate_tfrecord.py

```
