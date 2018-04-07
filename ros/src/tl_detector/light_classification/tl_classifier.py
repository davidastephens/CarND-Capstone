from styx_msgs.msg import TrafficLight

import glob
import os
import cv2
import sys

import tensorflow as tf
import numpy as np

# Traffic Light Detection Based on Tensorflow demo shown here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = os.path.join(FILE_PATH, "images")
OUTPUT_DIR_PATH = os.path.join(FILE_PATH, "images_out")

from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = os.path.join(FILE_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
NUM_CLASSES = 90


class TLClassifier(object):
    def __init__(self):

        # Load tensorflow graph into memory
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.num = self.graph.get_tensor_by_name('num_detections:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.session = tf.Session(graph=self.graph, config=config)

    def run_inference_for_single_image(self, image):
        with self.graph.as_default():
            # Run inference
            number, boxes, scores, classes = self.session.run([self.num, self.boxes, self.scores, self.classes],
                                   feed_dict={self.image_tensor: np.expand_dims(image, 0)})
        number = number[0]
        boxes = boxes[0]
        scores = scores[0]
        classes = classes[0]
        return number, boxes, scores, classes

    @staticmethod
    def filter_output_for_class(number, boxes, scores, classes, cls=10):
        mask = classes == cls
        return number, boxes[mask], scores[mask], classes[mask]


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        return TrafficLight.UNKNOWN


if __name__ == '__main__':
    image_files = glob.glob(os.path.join(DIR_PATH, '*.*'))

    classifier = TLClassifier()
    for image_file in image_files:
        img = cv2.imread(image_file)
        _, filename = os.path.split(image_file)
        number, boxes, scores, classes = classifier.run_inference_for_single_image(img)
        number, boxes, scores, classes = classifier.filter_output_for_class(number, boxes, scores, classes)
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            boxes,
            classes,
            scores,
            use_normalized_coordinates=True,
            min_score_thresh=0.3,
            agnostic_mode=True,
            line_thickness=8)
        cv2.imwrite(os.path.join(OUTPUT_DIR_PATH, filename), img)

