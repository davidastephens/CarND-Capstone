#!/usr/bin/env python
"""
Sample script to receive traffic light labels and images
of the Bosch Small Traffic Lights Dataset.
Example usage:
    python read_label_file.py input_yaml [test/train/all]
"""

import os
import sys
import yaml
import csv

def label_to_class(label):
    if label == 'Red':
        return 1
    if label == 'RedLeft':
        return 1
    if label == 'RedRight':
        return 1
    if label == 'RedStraight':
        return 1
    if label == 'RedStraightLeft':
        return 1
    if label == 'GreenLeft':
        return 3
    if label == 'GreenRight':
        return 3
    if label == 'GreenStraight':
        return 3
    if label == 'GreenStraightLeft':
        return 3
    if label == 'GreenStraightRight':
        return 3
    if label == 'Green':
        return 3
    if label == 'Yellow':
        return 2
    if label == 'off':
        return 4
    print(label)


def get_all_labels(input_yaml, kind='test'):
    """ Gets all labels within label file
    """
    images = yaml.load(open(input_yaml, 'rb').read())
    print(len(images))
    with open(kind+'.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['filename', 'class', 'xmin', 'ymin', 'xmax','ymax'])
        for i in range(len(images)):
            filename = images[i]['filename']
            for box in images[i]['annotations']:
                writer.writerow([filename, label_to_class(box['class']), box['xmin'], box['ymin'], box['xmin'] + box['x_width'], box['ymin'] + box['y_height']])
    return images


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    get_all_labels(sys.argv[1], kind=sys.argv[2])
