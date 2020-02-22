import cv2
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np
import FLSpegtransfer.utils.CmnUtil as U

class dvrkGraspingDetection(threading.Thread):

    def __init__(self, interval_ms=10, video_recording=False):
        # data members
        self.__bridge = CvBridge()
        self.__img_raw_left = []
        self.__img_raw_right = []
        self.__actuator_current_measured_PSM1 = []
        self.__actuator_current_measured_PSM2 = []
        self.__state_jaw_current_PSM1 = []
        self.__state_jaw_current_PSM2 = []

        # threading
        threading.Thread.__init__(self)
        self.__interval_ms = interval_ms
        self.__stop_flag = False

        # subscriber
        self.__sub_list = [rospy.Subscriber('/endoscope/left/image_raw/compressed', CompressedImage, self.__img_raw_left_cb),
                           rospy.Subscriber('/endoscope/right/image_raw/compressed', CompressedImage, self.__img_raw_right_cb),
                           rospy.Subscriber('/dvrk/PSM1/io/actuator_current_measured', JointState, self.__actuator_current_measured_PSM1_cb),
                           rospy.Subscriber('/dvrk/PSM2/io/actuator_current_measured', JointState, self.__actuator_current_measured_PSM2_cb),
                           rospy.Subscriber('/dvrk/PSM1/state_jaw_current', JointState, self.__state_jaw_current_PSM1_cb),
                           rospy.Subscriber('/dvrk/PSM2/state_jaw_current', JointState, self.__state_jaw_current_PSM2_cb)]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('detect_grasping_node', anonymous=True, log_level=rospy.WARN)
            self.rate = rospy.Rate(1000.0 / self.__interval_ms)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        # video recording
        self.__video_recording = video_recording
        if self.__video_recording is True:
            fps = 1000.0/self.__interval_ms
            fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            self.__img_width = 640
            self.__img_height = 360
            self.out = cv2.VideoWriter('video_recorded.avi', fcc, fps, (self.__img_width, self.__img_height))
            print ("start recording")

        # initializing variables
        self.__jaw1_curr_PSM1_prev = 0
        self.__jaw2_curr_PSM1_prev = 0
        self.__jaw1_curr_PSM2_prev = 0
        self.__jaw2_curr_PSM2_prev = 0

        self.start()
        rospy.spin()

    def start(self):
        self.__stop_flag = False
        self.__thread = threading.Thread(target=self.run, args=(lambda: self.__stop_flag,))
        self.__thread.daemon = True
        self.__thread.start()

    def stop(self):
        self.__stop_flag = True

    def run(self, stop):
        while True:
            # Resizing images
            img1 = cv2.resize(self.__img_raw_left, (self.__img_width, self.__img_height))
            img2 = cv2.resize(self.__img_raw_right, (self.__img_width, self.__img_height))

            # Low pass filtering
            fc = 1  # (Hz)
            dt = 0.01  # (ms)
            jaw1_curr_PSM1 = U.LPF(self.__actuator_current_measured_PSM1[5], self.__jaw1_curr_PSM1_prev, fc, dt)
            self.__jaw1_curr_PSM1_prev = jaw1_curr_PSM1
            jaw2_curr_PSM1 = U.LPF(self.__actuator_current_measured_PSM1[6], self.__jaw2_curr_PSM1_prev, fc, dt)
            self.__jaw2_curr_PSM1_prev = jaw2_curr_PSM1
            jaw1_curr_PSM2 = U.LPF(self.__actuator_current_measured_PSM2[5], self.__jaw1_curr_PSM2_prev, fc, dt)
            self.__jaw1_curr_PSM2_prev = jaw1_curr_PSM2
            jaw2_curr_PSM2 = U.LPF(self.__actuator_current_measured_PSM2[6], self.__jaw2_curr_PSM2_prev, fc, dt)
            self.__jaw2_curr_PSM2_prev = jaw2_curr_PSM2

            # Current thresholding
            self.__threshold_PSM1 = 0.3  # (A)
            self.__threshold_PSM2 = 0.25  # (A)
            if self.__state_jaw_current_PSM1[0] <= 0:
                if jaw1_curr_PSM1 - jaw2_curr_PSM1 > self.__threshold_PSM1:
                    cv2.circle(img1, (540, 300), 30, (0, 255, 0), -1)
                else:
                    cv2.circle(img1, (540, 300), 30, (0, 0, 255), -1)
            else:
                cv2.circle(img1, (540, 300), 30, (0, 0, 255), -1)

            if self.__state_jaw_current_PSM2[0] <= 0:
                if jaw1_curr_PSM2 - jaw2_curr_PSM2 > self.__threshold_PSM2:
                    cv2.circle(img1, (100, 300), 30, (0, 255, 0), -1)
                else:
                    cv2.circle(img1, (100, 300), 30, (0, 0, 255), -1)
            else:
                cv2.circle(img1, (100, 300), 30, (0, 0, 255), -1)

            cv2.imshow("original1", img1)

            if self.__video_recording is True:
                self.out.write(img1)
            self.rate.sleep()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.__video_recording is True:
                    print ("finishing recording")
                    self.out.release()
                cv2.destroyAllWindows()
                stop()
                break

    def __img_raw_left_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_left = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_left = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __img_raw_right_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_right = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_right = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
    dg = dvrkGraspingDetection(interval_ms=10, video_recording=True)