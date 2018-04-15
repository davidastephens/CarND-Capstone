#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial import KDTree
import numpy as np

import datetime
import os


from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

# Variables for saving images
SAVE_IMAGES = False
SECONDS_BETWEEN_IMAGES = 3
DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "light_classification", "images")


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.lights_2d = None
        self.lights_tree = None
        simulator = rospy.get_param("simulation", True)
        rospy.logwarn("Simulator: {}".format(simulator))
        self.simulator = simulator
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(simulator)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.last_image_save_time = datetime.datetime.now()
        self.image_count = 0
        # Run one inference to get the gpu going
        self.light_classifier.get_classification(np.zeros((640,480,3), np.float64))

        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # Use same method as Waypoint updator to create a KDTree
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if not self.lights_2d:
            self.lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y] for light in self.lights]
            self.lights_tree = KDTree(self.lights_2d)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        # process every 3rd image
        self.image_count += 1
        if self.image_count % 3 != 0:
            return

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            rospy.logwarn("State is: {}".format(state))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoint_tree:
            closest_idx = self.waypoint_tree.query([x, y], 1)[1]
            return closest_idx

        return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if (not self.has_image):
            self.prev_light_loc = None
            return False
        
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        if SAVE_IMAGES:
            now = datetime.datetime.now()
            time_delta = now - self.last_image_save_time
            if time_delta > datetime.timedelta(seconds=SECONDS_BETWEEN_IMAGES):
                self.last_image_save_time = now
                return self.light_classifier.get_classification(cv_image, save=True)

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if (self.pose):
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            closest_car_idx = self.get_closest_waypoint(x, y)

            if self.lights_tree:
                closest_light_idx = self.lights_tree.query([x, y], 1)[1]
                closest_stopline_location = stop_line_positions[closest_light_idx]
                closest_stopline_idx = self.get_closest_waypoint(*closest_stopline_location)
                diff = closest_stopline_idx - closest_car_idx
                if diff < 0:
                    closest_light_idx += 1
                    closest_stopline_location = stop_line_positions[closest_light_idx]
                    closest_stopline_idx = self.get_closest_waypoint(*closest_stopline_location)
        if not self.simulator and not self.pose:
            closest_light_idx = 0
            closest_stopline_idx = 0

        if closest_light_idx is not None:
            state = self.get_light_state(self.lights[closest_light_idx])
            return closest_stopline_idx, state

        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
