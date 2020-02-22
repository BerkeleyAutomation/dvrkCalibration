import cv2
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np
import FLSpegtransfer.utils.CmnUtil as U

class dvrkGraspingDetection(threading.Thread):

    def __init__(self, threshold_PSM1=0.3, threshold_PSM2=0.2):
        # initializing variables
        self.threshold_PSM1 = threshold_PSM1  # (A)
        self.threshold_PSM2 = threshold_PSM2  # (A)
        self.__jaw1_curr_PSM1_prev = 0
        self.__jaw2_curr_PSM1_prev = 0
        self.__jaw1_curr_PSM2_prev = 0
        self.__jaw2_curr_PSM2_prev = 0

        # data members
        self.__bridge = CvBridge()
        self.__actuator_current_measured_PSM1 = []
        self.__actuator_current_measured_PSM2 = []
        self.__state_jaw_current_PSM1 = []
        self.__state_jaw_current_PSM2 = []
        self.grasp_detected_PSM1 = False
        self.grasp_detected_PSM2 = False

        # subscriber
        self.__sub_list = [rospy.Subscriber('/dvrk/PSM1/io/actuator_current_measured', JointState, self.__actuator_current_measured_PSM1_cb),
                           rospy.Subscriber('/dvrk/PSM2/io/actuator_current_measured', JointState, self.__actuator_current_measured_PSM2_cb),
                           rospy.Subscriber('/dvrk/PSM1/state_jaw_current', JointState, self.__state_jaw_current_PSM1_cb),
                           rospy.Subscriber('/dvrk/PSM2/state_jaw_current', JointState, self.__state_jaw_current_PSM2_cb)]

        # create node
        self.__interval_ms = 10
        if not rospy.get_node_uri():
            rospy.init_node('grasping_detection_node', anonymous=True, log_level=rospy.WARN)
            self.rate = rospy.Rate(1000.0 / self.__interval_ms)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        # threading
        threading.Thread.__init__(self)
        self.thread = threading.Thread(target=self.loop)
        self.thread.daemon = True
        self.thread.start()

    def loop(self):
        while True:
            # Low pass filtering
            fc = 1  # (Hz)
            dt = self.__interval_ms*0.001  # (sec)
            jaw1_curr_PSM1 = U.LPF(self.__actuator_current_measured_PSM1[5], self.__jaw1_curr_PSM1_prev, fc, dt)
            self.__jaw1_curr_PSM1_prev = jaw1_curr_PSM1
            jaw2_curr_PSM1 = U.LPF(self.__actuator_current_measured_PSM1[6], self.__jaw2_curr_PSM1_prev, fc, dt)
            self.__jaw2_curr_PSM1_prev = jaw2_curr_PSM1
            jaw1_curr_PSM2 = U.LPF(self.__actuator_current_measured_PSM2[5], self.__jaw1_curr_PSM2_prev, fc, dt)
            self.__jaw1_curr_PSM2_prev = jaw1_curr_PSM2
            jaw2_curr_PSM2 = U.LPF(self.__actuator_current_measured_PSM2[6], self.__jaw2_curr_PSM2_prev, fc, dt)
            self.__jaw2_curr_PSM2_prev = jaw2_curr_PSM2

            # Current thresholding
            if (self.__state_jaw_current_PSM1[0] <= 0) and (jaw1_curr_PSM1 - jaw2_curr_PSM1 > self.threshold_PSM1):
                self.grasp_detected_PSM1 = True
            else:
                self.grasp_detected_PSM1 = False

            if (self.__state_jaw_current_PSM2[0] <= 0) and (jaw1_curr_PSM2 - jaw2_curr_PSM2 > self.threshold_PSM2):
                self.grasp_detected_PSM2 = True
            else:
                self.grasp_detected_PSM2 = False

            self.rate.sleep()

    def __actuator_current_measured_PSM1_cb(self, data):
        joint = np.array(0, dtype=np.float)
        joint.resize(len(data.position))
        joint.flat[:] = data.position
        self.__actuator_current_measured_PSM1 = list(joint)

    def __actuator_current_measured_PSM2_cb(self, data):
        joint = np.array(0, dtype=np.float)
        joint.resize(len(data.position))
        joint.flat[:] = data.position
        self.__actuator_current_measured_PSM2 = list(joint)

    def __state_jaw_current_PSM1_cb(self, data):
        joint = np.array(0, dtype=np.float)
        joint.resize(len(data.position))
        joint.flat[:] = data.position
        self.__state_jaw_current_PSM1 = list(joint)

    def __state_jaw_current_PSM2_cb(self, data):
        joint = np.array(0, dtype=np.float)
        joint.resize(len(data.position))
        joint.flat[:] = data.position
        self.__state_jaw_current_PSM2 = list(joint)

if __name__ == '__main__':
    GD = dvrkGraspingDetection(threshold_PSM1=0.3, threshold_PSM2=0.2)
    import time
    while True:
        print GD.grasp_detected_PSM1, GD.grasp_detected_PSM2
        time.sleep(0.5)