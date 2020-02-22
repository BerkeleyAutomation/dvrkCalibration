import numpy as np
import cv2
import ros_numpy
import rospy
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, PointCloud2

from FLSpegtransfer.vision.BlockDetection import BlockDetection
from FLSpegtransfer.vision.MappingC2R import MappingC2R
from FLSpegtransfer.motion.dvrkBlockTransfer import dvrkBlockTransfer
import FLSpegtransfer.utils.CmnUtil as U

class FLSPegTransfer():
    def __init__(self):
        # import other modules
        row_board = 6
        col_board = 8
        filename = 'calibration_files/mapping_table_PSM1'
        self.__mapping1 = MappingC2R(filename, row_board, col_board)
        filename = 'calibration_files/mapping_table_PSM2'
        self.__mapping2 = MappingC2R(filename, row_board, col_board)
        self.__block_detection = BlockDetection()
        self.__dvrk = dvrkBlockTransfer()

        # data members
        self.__bridge = CvBridge()
        self.__img_color = []
        self.__img_depth = []
        self.__points_list = []
        self.__points_ros_msg = PointCloud2()

        self.__moving_blocks_left2right_flag = False
        self.__moving_blocks_right2left_flag = False

        # ROS subscriber
        rospy.Subscriber('/zivid_camera/color/image_color/compressed', CompressedImage, self.__img_color_cb)
        rospy.Subscriber('/zivid_camera/depth/image_raw', Image, self.__img_depth_cb)
        # rospy.Subscriber('/zivid_camera/points', PointCloud2, self.__pcl_cb)  # not used in this time

        # create ROS node
        if not rospy.get_node_uri():
            rospy.init_node('Image_pipeline_node', anonymous=True, log_level=rospy.WARN)
            print ("ROS node initialized")
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        self.interval_ms = 300
        self.rate = rospy.Rate(1000.0 / self.interval_ms)
        self.main()

    def __img_color_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                img_raw = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                img_raw = self.__bridge.imgmsg_to_cv2(data, "bgr8")
            self.__img_color = self.__img_crop(img_raw)
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def __img_depth_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                img_raw = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                img_raw = self.__bridge.imgmsg_to_cv2(data, "32FC1")
            self.__img_depth = self.__img_crop(img_raw)
        except CvBridgeError as e:
            print(e)

    def __pcl_cb(self, data):
        pc = ros_numpy.numpify(data)
        points = np.zeros((pc.shape[0], pc.shape[1], 3))
        points[:, :, 0] = pc['x']
        points[:, :, 1] = pc['y']
        points[:, :, 2] = pc['z']
        self.__points_list = points

    def __img_crop(self, img):
        # Image cropping
        x = 650; w = 520
        y = 100; h = 400
        cropped = img[y:y + h, x:x + w]
        return cropped

    def transform_img2robot(self, final_grasping_pose, which_arm):
        if which_arm == 'PSM1':
            pts_robot = np.array([self.__mapping1.transform_pixel2robot(gp[2:]) for gp in final_grasping_pose])
        elif which_arm == 'PSM2':
            pts_robot = np.array([self.__mapping2.transform_pixel2robot(gp[2:]) for gp in final_grasping_pose])
        rot_robot = np.array([[gp[1]] for gp in final_grasping_pose])
        return np.hstack((rot_robot, pts_robot))

    def move_block(self, grasping_pose1, placing_pose1, pick_number1, place_number1, grasping_pose2, placing_pose2, pick_number2, place_number2):
        pos_pick1 = grasping_pose1[pick_number1-1][1:]
        rot_pick1 = [-grasping_pose1[pick_number1-1][0], 0, 0]      # (-) is added because the direction of pose perception and motion is oppoiste.
        pos_place1 = placing_pose1[place_number1 - 1][1:]
        rot_place1 = [-placing_pose1[place_number1 - 1][0], 0, 0]

        pos_pick2 = grasping_pose2[pick_number2-1][1:]
        rot_pick2 = [-grasping_pose2[pick_number2-1][0], 0, 0]
        pos_place2 = placing_pose2[place_number2-1][1:]
        rot_place2 = [-placing_pose2[place_number2-1][0], 0, 0]

        # conversion to quaternion
        q_pick1 = U.euler_to_quaternion(rot_pick1)
        q_place1 = U.euler_to_quaternion(rot_place1)
        q_pick2 = U.euler_to_quaternion(rot_pick2)
        q_place2 = U.euler_to_quaternion(rot_place2)
        self.__dvrk.pickup(pos_pick1, q_pick1, pos_pick2, q_pick2, 'PSM2')
        self.__dvrk.place(pos_place1, q_place1, pos_place2, q_place2, 'PSM2')

    def main(self):
        try:
            while True:
                if self.__img_color == [] or self.__img_depth == []:
                    pass
                else:
                    # self.__dvrk.move_origin()
                    # time.sleep(0.3)
                    #
                    # img_cb = cv2.imread('img/img_color_checkerboard.png')
                    # print img_cb.shape
                    # print self.__img_color.shape
                    # mixed = img_cb + self.__img_color
                    # cv2.imshow("", mixed)
                    # key = cv2.waitKey(1) & 0xFF
                    # if key == ord('1'):
                    #     p = [[1, 0, 121, 196],[2, 0, 121, 196]]
                    #     # Move blocks from left to right
                    #     # Transform img points to robot coordinates
                    #     grasping_pose_robot2 = self.transform_img2robot(p, 'PSM2')
                    #     placing_pose_robot2 = self.transform_img2robot(p, 'PSM2')
                    #     self.move_block(grasping_pose_robot2, placing_pose_robot2, 1, 1, grasping_pose_robot2, placing_pose_robot2, 1, 1)

                    # Scanning
                    self.__dvrk.move_origin()
                    time.sleep(0.3)

                    # Perception output
                    peg_points, final_grasping_pose_left, final_grasping_pose_right, final_placing_pose, pegs_overlayed, blocks_overlayed = self.__block_detection.FLSPerception(self.__img_depth)
                    # print final_grasping_pose_left

                    cv2.imshow("img_color", self.__img_color)
                    cv2.imshow("masked_pegs", pegs_overlayed)
                    cv2.imshow("masked_blocks", blocks_overlayed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('1'):
                        self.__moving_blocks_left2right_flag = True
                        self.__moving_blocks_right2left_flag = False
                    elif key == ord('2'):
                        self.__moving_blocks_left2right_flag = False
                        self.__moving_blocks_right2left_flag = True
                    else:
                        self.__moving_blocks_left2right_flag = False
                        self.__moving_blocks_right2left_flag = False

                    if self.__moving_blocks_left2right_flag:
                        # Move blocks from left to right
                        # Transform img points to robot coordinates
                        grasping_pose_robot1 = self.transform_img2robot(final_grasping_pose_right, 'PSM1')
                        grasping_pose_robot2 = self.transform_img2robot(final_grasping_pose_left, 'PSM2')
                        placing_pose_robot1 = self.transform_img2robot(final_placing_pose, 'PSM1')
                        placing_pose_robot2 = self.transform_img2robot(final_placing_pose, 'PSM2')
                        self.move_block(grasping_pose_robot1, placing_pose_robot1, 2, 9, grasping_pose_robot2, placing_pose_robot2, 1, 8)
                    #     self.move_block(grasping_pose_robot1, placing_pose_robot1, 4, 12, grasping_pose_robot2, placing_pose_robot2, 3, 7)
                    #     self.move_block(grasping_pose_robot1, placing_pose_robot1, 6, 11, grasping_pose_robot2, placing_pose_robot2, 5, 10)
                    # if self.__moving_blocks_right2left_flag:
                    #     # Move blocks from right to left
                    #     # Transform img points to robot coordinates
                    #     grasping_pose_robot1 = self.transform_img2robot(final_grasping_pose_right, 'PSM1')
                    #     grasping_pose_robot2 = self.transform_img2robot(final_grasping_pose_left, 'PSM2')
                    #     placing_pose_robot1 = self.transform_img2robot(final_placing_pose, 'PSM1')
                    #     placing_pose_robot2 = self.transform_img2robot(final_placing_pose, 'PSM2')
                    #     self.move_block(grasping_pose_robot1, placing_pose_robot1, 11, 6, grasping_pose_robot2, placing_pose_robot2, 10, 5)
                    #     self.move_block(grasping_pose_robot1, placing_pose_robot1, 12, 4, grasping_pose_robot2, placing_pose_robot2, 7, 3)
                    #     self.move_block(grasping_pose_robot1, placing_pose_robot1, 9, 2, grasping_pose_robot2, placing_pose_robot2, 8, 1)

        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    FLSPegTransfer()