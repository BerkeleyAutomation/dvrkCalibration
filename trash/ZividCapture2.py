import sys
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2

def add_two_ints_client(x, y):
    rospy.wait_for_service('/zivid_camera/color/image_color/compressed')
    try:
        add_two_ints = rospy.ServiceProxy('/zivid_camera/color/image_color/compressed', CompressedImage)
        resp1 = add_two_ints(x, y)
        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

def usage():
    return "%s [x y]" % sys.argv[0]

if __name__ == "__main__":
    add_two_ints_client(1,2)