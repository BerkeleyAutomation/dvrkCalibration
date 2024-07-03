from FLSpegtransfer.vision.ZividCapture import ZividCapture
import cv2
import numpy as np

which_camera = "inclined"
zivid = ZividCapture(which_camera=which_camera)
zivid.start()

# while True:
#     image = zc.capture_2Dimage(color='BGR')
#     cv2.imshow("", image)
#     cv2.waitKey(1)

# check images
img_color, img_depth, img_point = zivid.capture_3Dimage(color='BGR')

# zivid.display_rgb(img_color, block=False)
# zivid.display_rgb(img_color, block=True)
# zivid.display_depthmap(img_point)
# zivid.display_pointcloud(img_point, img_color)

cv2.imwrite("color_" + which_camera + ".png", img_color)
np.save("img_color", img_color)
np.save("img_depth_" + which_camera, img_depth)
np.save("img_point_" + which_camera, img_point)
