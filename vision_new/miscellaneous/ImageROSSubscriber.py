import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import JointState, CompressedImage, Image


class ImageROSSubscriber:
    def __init__(self, image_type='raw', which_camera='alliedvision'):
        self.image_type = image_type

        # Create node
        if not rospy.get_node_uri():
            rospy.init_node('image_subscriber_node', anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        # data members
        self.img_left = []
        self.img_right = []
        self.bridge = CvBridge()

        if which_camera == 'alliedvision':
            # ROS
            if image_type == 'raw':
                rospy.Subscriber('/av/img_left', Image, self.__get_image_left_cb)
                rospy.Subscriber('/av/img_right', Image, self.__get_image_right_cb)
            elif image_type == 'compressed':
                rospy.Subscriber('/av/img_left', CompressedImage, self.__get_image_left_cb)
                rospy.Subscriber('/av/img_right', CompressedImage, self.__get_image_right_cb)
            else:
                raise ValueError
        elif which_camera == 'endoscope':
            # ROS
            if image_type == 'raw':
                rospy.Subscriber('/endoscope/left/image_color', Image, self.__get_image_left_cb)
                rospy.Subscriber('/endoscope/right/image_color', Image, self.__get_image_right_cb)
            elif image_type == 'compressed':
                rospy.Subscriber('/endoscope/left/image_color/compressed', CompressedImage, self.__get_image_left_cb)
                rospy.Subscriber('/endoscope/right/image_color/compressed', CompressedImage, self.__get_image_right_cb)
            else:
                raise ValueError

    def __get_image_left_cb(self, data):
        if self.image_type == 'raw':
            self.img_left = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        elif self.image_type == 'compressed':
            self.img_left = self.bridge.compressed_imgmsg_to_cv2(data)
        else:
            raise ValueError

    def __get_image_right_cb(self, data):
        if self.image_type == 'raw':
            self.img_right = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        elif self.image_type == 'compressed':
            self.img_right = self.bridge.compressed_imgmsg_to_cv2(data)
        else:
            raise ValueError


if __name__ == '__main__':
    import cv2
    from dvrk.utils.ImgUtils import ImgUtils
    subs = ImageROSSubscriber(image_type='compressed', which_camera='alliedevision')
    while True:
        img_left = subs.img_left
        img_right = subs.img_right
        if len(img_left) == 0 or len(img_right) == 0:
            pass
        else:
            # stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
            cv2.imshow("", img_left)
            cv2.waitKey(0)
