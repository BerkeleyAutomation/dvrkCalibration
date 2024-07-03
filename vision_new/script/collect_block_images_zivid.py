from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.motion.deprecated.dvrkMotionBridgeP import dvrkMotionBridgeP
import FLSpegtransfer.utils.CmnUtil as U
import sys
import os
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

zc = ZividCapture()
zc.start()

# check images
img_color, img_depth, img_point = zc.capture_3Dimage(color='BGR')
zc.display_rgb(img_color)
zc.display_depthmap(img_point)
zc.display_pointcloud(img_point, img_color)

dvrk = dvrkMotionBridgeP()
sampling = 100
for i in range(sampling):
    r1 = np.random.uniform(-40, 40)
    r2 = np.random.uniform(-40, 40)
    r3 = np.random.uniform(-40, 40)
    rot1 = [r1, r2, r3]
    q1 = U.euler_to_quaternion(rot1, unit='deg')
    jaw1 = [10 * np.pi / 180.]
    dvrk.set_pose(rot1=q1, jaw1=jaw1)
    img_color, img_depth, img_point = zc.capture_3Dimage(img_crop=False)

    os.makedirs("block_images/" + str(i), exist_ok=True)
    np.save("block_images/" + str(i) + "/color_dropping", img_color)
    np.save("block_images/" + str(i) + "/depth_dropping", img_depth)
    np.save("block_images/" + str(i) + "/point_dropping", img_point)
    color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
    cv2.imwrite("block_images/" + str(i) + "/color_dropping.png", color)