import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2

class BallDetection():
    def __init__(self):
        # data members
        self.__img_color = []
        self.__img_depth = []
        self.__img_point = []

        # thresholding value
        self.__masking_depth = [300, 830]
        # self.__lower_red = np.array([-30, 50, 50])
        # self.__upper_red = np.array([30, 255, 255])
        self.__lower_red = np.array([-30, 155, 84])
        self.__upper_red = np.array([30, 255, 255])
        self.__lower_blue = np.array([94, 80, 2])
        self.__upper_blue = np.array([126, 255, 255])
        self.__lower_green = np.array([25, 52, 72])
        self.__upper_green = np.array([102, 255, 255])
        # self.__lower_yellow = np.array([-30, 50, 50])
        # self.__upper_yellow = np.array([30, 255, 255])
        # self.__lower_white = np.array([0, 255, 255])
        # self.__upper_white = np.array([180, 255, 255])

        self.r1 = 0.025  # ball radius (m)
        self.r2 = 0.020
        self.r3 = 0.016
        self.L1 = 0.045  # ball1 ~ ball2 (m)
        self.L2 = 0.016  # ball2 ~ pitch (m)
        self.L3 = 0.0091  # pitch ~ yaw (m)
        self.L4 = 0.0102  # yaw ~ tip (m)

        # camera intrinsic parameters
        self.__D = [-0.2826650142669678, 0.42553916573524475, -0.0005135679966770113, -0.000839113024994731,
                    -0.5215581655502319]
        # self.__K = [[2776.604248046875, 0.0, 952.436279296875], [0.0, 2776.226318359375, 597.9248046875], [0.0, 0.0, 1.0]]
        # self.__K = [[2840.0, 0.0, 945], [0.0, 2890.0, 603], [0.0, 0.0, 1.0]]
        self.__K = [[2770, 0.0, 955], [0.0, 2775, 601], [0.0, 0.0, 1.0]]
        self.__R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.__P = [[2776.604248046875, 0.0, 952.436279296875], [0.0, 0.0, 2776.226318359375],
                    [597.9248046875, 0.0, 0.0], [0.0,
                                                 1.0, 0.0]]
        self.__fx = self.__K[0][0]
        self.__fy = self.__K[1][1]
        self.__cx = self.__K[0][2]
        self.__cy = self.__K[1][2]

        # crop parameters
        # self.__xcr = 710
        # self.__wcr = 520
        # self.__ycr = 450
        # self.__hcr = 400
        self.__xcr = 510
        self.__wcr = 920
        self.__ycr = 250
        self.__hcr = 800

    def img_crop(self, img_color, img_depth, img_point):
        color_cropped = img_color[self.__ycr:self.__ycr + self.__hcr, self.__xcr:self.__xcr + self.__wcr]
        depth_cropped = img_depth[self.__ycr:self.__ycr + self.__hcr, self.__xcr:self.__xcr + self.__wcr]
        point_cropped = img_point[self.__ycr:self.__ycr + self.__hcr, self.__xcr:self.__xcr + self.__wcr]
        return color_cropped, depth_cropped, point_cropped

    def pixel2world(self, x, y, depth):
        Xc = (x - self.__cx + self.__xcr) / self.__fx * depth
        Yc = (y - self.__cy + self.__ycr) / self.__fy * depth
        Zc = depth
        return Xc, Yc, Zc

    def world2pixel(self, Xc, Yc, Zc):
        x = self.__fx * Xc / Zc + self.__cx - self.__xcr
        y = self.__fy * Yc / Zc + self.__cy - self.__ycr
        return int(x), int(y)

    def overlay_ball(self, img, x,y,r):
        overlayed = np.copy(img)
        cv2.circle(overlayed, (x, y), r, (0, 255, 255), 2)
        cv2.circle(overlayed, (x, y), 4, (0, 255, 255), -1)
        return overlayed

    def overlay_tip(self, img, ptx, pty):
        overlayed = np.copy(img)
        cv2.circle(overlayed, (ptx, pty), 4, (0, 255, 0), 2)
        return overlayed

    def find_ball(self, img_color, img_depth, img_point, color, nb, radius):
        # Depth masking: thresholding by depth to find blocks & pegs
        mask_depth = cv2.inRange(img_depth, self.__masking_depth[0], self.__masking_depth[1])
        depth_masked = cv2.bitwise_and(img_color, img_color, mask=mask_depth)

        # Color masking
        hsv = cv2.cvtColor(depth_masked, cv2.COLOR_BGR2HSV)
        if color=='red':
            masked = cv2.inRange(hsv, self.__lower_red, self.__upper_red)
        elif color=='green':
            masked = cv2.inRange(hsv, self.__lower_green, self.__upper_green)
        elif color=='blue':
            masked = cv2.inRange(hsv, self.__lower_blue, self.__upper_blue)
        elif color=='yellow':
            masked = cv2.inRange(hsv, self.__lower_yellow, self.__upper_yellow)

        # Find contours in the mask and initialize the current (x, y) center of the ball
        cnts, _ = cv2.findContours(masked.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # only proceed if at least one contour was found
        pb = []
        pb_img = []
        if len(cnts) >= nb:
            # use the largest contour in the mask to compute the minimum enclosing circle and centroid
            # Positions in pixel coordinate
            for i in range(nb):
                ((cx, cy), r) = cv2.minEnclosingCircle(cnts[i])
                pb_img.append([int(cx), int(cy), int(r)])
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                depth = img_depth[int(cy), int(cx)] / 1000  # (m)
                if not np.isnan(depth):
                    pb.append(img_point[int(cy), int(cx)] + [0.0, 0.0, radius[i]])    # 3D position
        return pb, pb_img, masked

    # Get tool position from two ball positions
    def get_tool_position(self, pcb1, pcb2):
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        L4 = self.L4
        p_pitch = ((L1+L2)*pcb2-L2*pcb1)/L1
        p_tip = ((L1+L2+L3+L4)*pcb2-(L2+L3+L4)*pcb1)/L1
        return p_pitch, p_tip

    # Get tip orientation from three ball positions
    def get_orientation(self, pb1, pb2, pb3):
        pass

if __name__ == "__main__":
    from FLSpegtransfer.vision.ZividCapture import ZividCapture
    BD = BallDetection()
    zivid = ZividCapture()
    while True:
        try:
            zivid.capture_3Dimage()
            img_color, img_depth, img_point = BD.img_crop(zivid.image, zivid.depth, zivid.point)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
            img_color_org = np.copy(img_color)

            pb, pb_img, red_masked = BD.find_ball(img_color_org, img_depth, img_point, 'red', 2, [BD.r1, BD.r2])
            if len(pb) >= 2:
                _, pt = BD.get_tool_position(pb[0], pb[1])
                pt_img = BD.world2pixel(pt[0], pt[1], pt[2])

                img_color = BD.overlay_ball(img_color, pb_img[0][0], pb_img[0][1], pb_img[0][2])
                img_color = BD.overlay_ball(img_color, pb_img[1][0], pb_img[1][1], pb_img[1][2])
                img_color = BD.overlay_tip(img_color, pt_img[0], pt_img[1])

            pb3, pb3_img, blue_masked = BD.find_ball(img_color_org, img_depth, img_point, 'blue', 3, [BD.r3, BD.r3, BD.r3])
            if len(pb3) >= 3:
                img_color = BD.overlay_ball(img_color, pb3_img[0][0], pb3_img[0][1], pb3_img[0][2])
                img_color = BD.overlay_ball(img_color, pb3_img[1][0], pb3_img[1][1], pb3_img[1][2])
                img_color = BD.overlay_ball(img_color, pb3_img[2][0], pb3_img[2][1], pb3_img[2][2])

            pb4, pb4_img, green_masked = BD.find_ball(img_color_org, img_depth, img_point, 'green', 1, [BD.r3])
            if len(pb4) >= 1:
                img_color = BD.overlay_ball(img_color, pb4_img[0][0], pb4_img[0][1], pb4_img[0][2])

            cv2.imshow("images", img_color)
            # cv2.imshow("original", img_color)
            # cv2.imshow("depth_masked", depth_masked)
            cv2.imshow("red_masked", red_masked)
            cv2.imshow("blue_masked", blue_masked)
            cv2.imshow("green_masked", green_masked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        finally:
            pass