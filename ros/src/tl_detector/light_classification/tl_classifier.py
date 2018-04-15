from styx_msgs.msg import TrafficLight
import rospy

import glob
import os
import cv2
import datetime

import tensorflow as tf
import numpy as np

# Traffic Light Detection Based on Tensorflow demo shown here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = os.path.join(FILE_PATH, "images")
OUTPUT_DIR_PATH = os.path.join(FILE_PATH, "images_out")

from utils import visualization_utils as vis_util

MODEL_NAME = 'traffic_light_graph'
PATH_TO_CKPT_SIM = os.path.join(FILE_PATH, '..', '..', '..', '..', 'classifier', MODEL_NAME, 'frozen_inference_graph.pb')
MODEL_NAME = 'traffic_light_graph_real'
PATH_TO_CKPT_REAL = os.path.join(FILE_PATH, '..', '..', '..', '..', 'classifier', MODEL_NAME, 'frozen_inference_graph.pb')
NUM_CLASSES = 4

CATEGORY_INDEX ={1: {'id': 1, 'name': 'Red'},
                    2: {'id': 2, 'name': 'Yellow'},
                    3: {'id': 3, 'name': 'Green'},
                    4: {'id': 4, 'name': 'Unknown'}}  

class TLClassifier(object):
    def __init__(self, simulator):

        # Load tensorflow graph into memory
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if simulator:
            PATH_TO_CKPT = PATH_TO_CKPT_SIM
        else:
            PATH_TO_CKPT = PATH_TO_CKPT_REAL

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
        classes = classes[0].astype(int)
        return number, boxes, scores, classes

    @staticmethod
    def filter_output_for_class(number, boxes, scores, classes, cls=10):
        mask = classes == cls
        return number, boxes[mask], scores[mask], classes[mask]

    def get_classification(self, image, save=False):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        number, boxes, scores, classes = self.run_inference_for_single_image(image)
        predict, threshold = classes[0], scores[0]
        if save:
            rospy.logwarn('Saving vizualization')
            vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes,
            scores,
            category_index=CATEGORY_INDEX,
            use_normalized_coordinates=True,
            min_score_thresh=0.50,
            agnostic_mode=False,
            line_thickness=8)
            filename=str(predict) + '_' + str(datetime.datetime.now()) + '.jpg'
            cv2.imwrite(os.path.join(OUTPUT_DIR_PATH, filename),image)

        if threshold < 0.3:
            return TrafficLight.UNKNOWN
        if predict == 1:
            #rospy.logwarn('Red Light')
            return TrafficLight.RED
        elif predict == 2:
            return TrafficLight.YELLOW
        elif predict == 3:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN

if __name__ == '__main__':
    image_files = glob.glob(os.path.join(DIR_PATH, '*.*'))

    classifier = TLClassifier()


    for image_file in image_files:
        img = cv2.imread(image_file)
        _, filename = os.path.split(image_file)
        number, boxes, scores, classes = classifier.run_inference_for_single_image(img)
        print("Classes: {}".format(classes))
        print("Scores: {}".format(scores))
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            boxes,
            classes,
            scores,
            category_index=category_index,
            use_normalized_coordinates=True,
            min_score_thresh=0.20,
            agnostic_mode=False,
            line_thickness=8)
        cv2.imwrite(os.path.join(OUTPUT_DIR_PATH, filename), img)

