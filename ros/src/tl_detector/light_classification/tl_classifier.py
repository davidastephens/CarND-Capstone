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

    def run_inference_for_single_image(self, image):

        with self.graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

        return output_dict

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
        output_dict = classifier.run_inference_for_single_image(img)
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            use_normalized_coordinates=True,
            agnostic_mode=True,
            line_thickness=8)
        cv2.imwrite(os.path.join(OUTPUT_DIR_PATH, filename), img)
