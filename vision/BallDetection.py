import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import FLSpegtransfer.utils.CmnUtil as U
root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'

class BallDetection():
    def __init__(self):
        # data members
        self.__img_color = []
        self.__img_depth = []
        self.__img_point = []

        # thresholding value
        self.__masking_depth = [300, 950]
        self.__lower_red = np.array([0-20, 130, 40])
        self.__upper_red = np.array([0+20, 255, 255])
        self.__lower_green = np.array([60-20, 130, 40])
        self.__upper_green = np.array([60+20, 255, 255])
        self.__lower_blue = np.array([120-20, 130, 40])
        self.__upper_blue = np.array([120+20, 255, 255])
        self.__lower_yellow = np.array([30-10, 130, 60])
        self.__upper_yellow = np.array([30+10, 255, 255])

        # dimension of tool
        self.d = 35       # length of coordinate (mm)
        self.Lbb = 0.050  # ball1 ~ ball2 (m)
        self.Lbp = 0.017  # ball2 ~ pitch (m)
        self.L1 = 0.4318  # Rcc (m)
        self.L2 = 0.4162  # tool
        self.L3 = 0.0091  # pitch ~ yaw (m)
        self.L4 = 0.0102  # yaw ~ tip (m)

        # Transform from camera to robot
        self.Trc = np.load(root+'experiment/1_rigid_transformation/Trc.npy')
        self.Rrc = self.Trc[:3, :3]
        self.trc = self.Trc[:3, 3]

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

    def world2pixel(self, Xc, Yc, Zc, Rc=0):
        x = self.__fx * Xc / Zc + self.__cx - self.__xcr
        y = self.__fy * Yc / Zc + self.__cy - self.__ycr
        r = (self.__fx+self.__fy)/2 * Rc / Zc
        return int(x), int(y), int(r)

    def overlay_balls(self, img_color, pbs):
        overlayed = img_color.copy()
        for pb in pbs:
            if pb==[]:
                pass
            else:
                pb_img = self.world2pixel(pb[0], pb[1], pb[2], pb[3])
                cv2.circle(overlayed, (pb_img[0], pb_img[1]), pb_img[2], (0, 255, 255), 2)
                cv2.circle(overlayed, (pb_img[0], pb_img[1]), 3, (0, 255, 255), -1)
        return overlayed

    def overlay_tool(self, img_color, joint_angles, color):
        q1,q2,q3,q4,q5,q6 = joint_angles
        # 3D points w.r.t camera frame
        pb = self.Rrc.T.dot(np.array([0,0,0])-self.trc)*1000    # base position
        p5 = self.fk_position(q1,q2,q3,q4,q5,q6,L1=self.L1,L2=self.L2,L3=0,L4=0)
        p5 = self.Rrc.T.dot(np.array(p5)-self.trc)*1000  # pitch axis
        p6 = self.fk_position(q1,q2,q3,q4,q5,q6,L1=self.L1,L2=self.L2,L3=self.L3,L4=0)
        p6 = self.Rrc.T.dot(np.array(p6)-self.trc)*1000  # yaw axis
        p7 = self.fk_position(q1,q2,q3,q4,q5,q6,L1=self.L1,L2=self.L2,L3=self.L3,L4=self.L4+0.005)
        p7 = self.Rrc.T.dot(np.array(p7)-self.trc)*1000  # tip

        pb_img = self.world2pixel(pb[0], pb[1], pb[2])
        p5_img = self.world2pixel(p5[0], p5[1], p5[2])
        p6_img = self.world2pixel(p6[0], p6[1], p6[2])
        p7_img = self.world2pixel(p7[0], p7[1], p7[2])

        overlayed = img_color.copy()
        self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0,255,0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
        self.drawline(overlayed, p5_img[0:2], p6_img[0:2], (0,255,0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p6_img[0:2], 2, color, 2)
        self.drawline(overlayed, p6_img[0:2], p7_img[0:2], (0,255,0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p7_img[0:2], 2, color, 2)
        return overlayed

    def overlay_tool_position(self, img_color, joint_angles, color):
        q1, q2, q3 = joint_angles
        # 3D points w.r.t camera frame
        pb = self.Rrc.T.dot(np.array([0, 0, 0]) - self.trc) * 1000  # base position
        p5 = self.fk_position(q1, q2, q3, 0, 0, 0, L1=self.L1, L2=self.L2, L3=0, L4=0)
        p5 = self.Rrc.T.dot(np.array(p5) - self.trc) * 1000  # pitch axis

        pb_img = self.world2pixel(pb[0], pb[1], pb[2])
        p5_img = self.world2pixel(p5[0], p5[1], p5[2])

        overlayed = img_color.copy()
        self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0, 255, 0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
        return overlayed

    def drawline(self, img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            pts.append((x, y))

        if style == 'dotted':
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        elif style == 'dashed':
            st = pts[0]
            ed = pts[0]
            i = 0
            for p in pts:
                st = ed
                ed = p
                if i % 2 == 1:
                    cv2.line(img, st, ed, color, thickness)
                i += 1

    def mask_image(self, img_color, img_depth, img_point, color):
        # Depth masking
        mask_depth = cv2.inRange(img_depth, self.__masking_depth[0], self.__masking_depth[1])
        depth_masked = cv2.bitwise_and(img_color, img_color, mask=mask_depth)

        # Color masking
        hsv = cv2.cvtColor(depth_masked, cv2.COLOR_BGR2HSV)
        if color == 'red':
            masked = cv2.inRange(hsv, self.__lower_red, self.__upper_red)
        elif color == 'green':
            masked = cv2.inRange(hsv, self.__lower_green, self.__upper_green)
        elif color == 'blue':
            masked = cv2.inRange(hsv, self.__lower_blue, self.__upper_blue)
        elif color == 'yellow':
            masked = cv2.inRange(hsv, self.__lower_yellow, self.__upper_yellow)
        return masked

    def find_balls(self, img_color, img_depth, img_point):
        red_masked = self.mask_image(img_color, img_depth, img_point, 'red')
        green_masked = self.mask_image(img_color, img_depth, img_point, 'green')
        blue_masked = self.mask_image(img_color, img_depth, img_point, 'blue')
        yellow_masked = self.mask_image(img_color, img_depth, img_point, 'yellow')

        # cv2.imshow("red", red_masked)
        # cv2.imshow("green", green_masked)
        # cv2.imshow("blue", blue_masked)
        # cv2.imshow("yellow", yellow_masked)
        # cv2.waitKey(0)

        masked_img = [lambda:red_masked, lambda:red_masked, lambda:red_masked, lambda:green_masked, lambda:blue_masked, lambda:yellow_masked]
        radius = [12.0, 10.0, 8.0, 8.0, 8.0, 8.0]    # (mm)
        pb = []
        for i in range(len(masked_img)):
            # Find contours in the mask and initialize the current (x, y) center of the ball
            cnts, _ = cv2.findContours(masked_img[i](), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            if len(cnts)==0:
                pb.append([])
            else:
                # Find 3D points of a ball
                # Get the pixel coordinates inside the contour
                infilled = np.zeros(np.shape(img_color), np.uint8)
                cv2.drawContours(infilled, [cnts[0]], 0, (255, 255, 255), -1)
                infilled = cv2.cvtColor(infilled, cv2.COLOR_BGR2GRAY)
                infilled_inv = cv2.bitwise_not(infilled)
                ball_masked = cv2.bitwise_and(masked_img[i](), masked_img[i](), mask=infilled)

                # Get the point clouds
                args = np.argwhere(ball_masked == 255)
                points_ball = np.array([img_point[p[0], p[1]] for p in args])

                # Saving images
                # black = np.zeros_like(img_color)
                # for p in args:
                #     black[p[0], p[1]] = img_color[p[0], p[1]]
                # cv2.imshow("masked", black)
                # cv2.imwrite("masked.png", black)
                # cv2.waitKey(0)

                # Linear regression to fit the circle into the point cloud
                xc, yc, zc, rc = self.fit_circle_3d(points_ball[:, 0], points_ball[:, 1], points_ball[:, 2])
                if radius[i]-2 < rc < radius[i]+2:
                    pb.append([xc, yc, zc, rc])
                else:
                    pb.append([])

                # Masking the detected region
                red_masked = cv2.bitwise_and(masked_img[i](), masked_img[i](), mask=infilled_inv)
        return pb   # (mm)

    def fit_circle_3d(self, x, y, z, w=[]):
        A = np.array([x, y, z, np.ones(len(x))]).T
        b = x ** 2 + y ** 2 + z ** 2

        # Modify A,b for weighted least squares
        if len(w) == len(x):
            W = np.diag(w)
            A = np.dot(W, A)
            b = np.dot(W, b)

        # Solve by method of least squares
        c = np.linalg.lstsq(A, b, rcond=None)[0]

        # Get circle parameters from solution c
        xc = c[0] / 2
        yc = c[1] / 2
        zc = c[2] / 2
        r = np.sqrt(c[3] + xc ** 2 + yc ** 2 + zc ** 2)
        return xc, yc, zc, r

    # Get tool position of the pitch axis from two ball positions w.r.t. camera base coordinate
    def find_tool_position(self, pb1, pb2):
        pb1 = np.asarray(pb1[0:3], dtype=float)
        pb2 = np.asarray(pb2[0:3], dtype=float)
        p_pitch = ((self.Lbb+self.Lbp)*pb2-self.Lbp*pb1)/self.Lbb
        return p_pitch    # (mm), w.r.t. camera base coordinate

    # Get orientation from three ball positions w.r.t. robot base coordinate
    def find_tool_orientation(self, pbr, pbg, pbb, pby):
        pbr = np.array(pbr) # red
        pbg = np.array(pbg) # green
        pbb = np.array(pbb) # blue
        pby = np.array(pby) # yellow

        pts1 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])  # base pose w.r.t. robot coordinate
        # normalized direction vectors
        if pbr.size==0:   # red ball is occluded
            pborg = (pbg + pby)/2
            v1 = self.Rrc.dot((pborg-pbb)[0:3])
            v1 = v1/np.linalg.norm(v1)
            v2 = self.Rrc.dot((pbg-pby)[0:3])
            v2 = v2/np.linalg.norm(v2)
            v3 = np.cross(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        elif pbg.size==0:   # green ball is occluded
            pborg = (pbr + pbb) / 2
            v1 = self.Rrc.dot((pbr - pbb)[0:3])
            v1 = v1 / np.linalg.norm(v1)
            v2 = self.Rrc.dot((pborg - pby)[0:3])
            v2 = v2 / np.linalg.norm(v2)
            v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif pbb.size==0:   # blue ball is occluded
            pborg = (pbg + pby) / 2
            v1 = self.Rrc.dot((pbr - pborg)[0:3])
            v1 = v1 / np.linalg.norm(v1)
            v2 = self.Rrc.dot((pbg - pby)[0:3])
            v2 = v2 / np.linalg.norm(v2)
            v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif pby.size==0:   # yellow ball is occluded
            pborg = (pbr + pbb) / 2
            v1 = self.Rrc.dot((pbr - pbb)[0:3])
            v1 = v1 / np.linalg.norm(v1)
            v2 = self.Rrc.dot((pbg - pborg)[0:3])
            v2 = v2 / np.linalg.norm(v2)
            v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        else:
            pborg = (pbg + pby) / 2
            v1 = self.Rrc.dot((pbr - pborg)[0:3])
            v1 = v1 / np.linalg.norm(v1)
            v2 = self.Rrc.dot((pbg - pby)[0:3])
            v2 = v2 / np.linalg.norm(v2)
            v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        pts2 = np.array([v1, v2, v3])
        pts1 = pts1.T
        pts2 = pts2.T
        Rb = pts2.dot(np.linalg.inv(pts1))
        return Rb   # w.r.t. robot base coordinate

    def fk_position(self, q1, q2, q3, q4, q5, q6, L1=0, L2=0, L3=0, L4=0):
        xtip = L2 * np.cos(q2)*np.sin(q1) - L1*np.cos(q2) * np.sin(q1) + q3 * np.cos(q2) * np.sin(q1) + L3 * np.cos(
            q2) * np.cos(q5) * np.sin(
            q1) + L4 * np.cos(q1) * np.cos(q4) * np.sin(q6) - L3 * np.cos(q1) * np.sin(q4) * np.sin(q5) + L4 * np.cos(
            q2) * np.cos(q5) * np.cos(
            q6) * np.sin(q1) - L4 * np.cos(q1) * np.cos(q6) * np.sin(q4) * np.sin(q5) - L3 * np.cos(q4) * np.sin(
            q1) * np.sin(q2) * np.sin(
            q5) - L4 * np.sin(q1) * np.sin(q2) * np.sin(q4) * np.sin(q6) - L4 * np.cos(q4) * np.cos(q6) * np.sin(
            q1) * np.sin(q2) * np.sin(q5)

        ytip = L1 * np.sin(q2) - L2 * np.sin(q2) - q3 * np.sin(q2) - L3 * np.cos(q5) * np.sin(q2) - L3 * np.cos(
            q2) * np.cos(q4) * np.sin(
            q5) - L4 * np.cos(q5) * np.cos(q6) * np.sin(q2) - L4 * np.cos(q2) * np.sin(q4) * np.sin(q6) - L4 * np.cos(
            q2) * np.cos(q4) * np.cos(
            q6) * np.sin(q5)

        ztip = L1 * np.cos(q1) * np.cos(q2) - L2 * np.cos(q1) * np.cos(q2) - q3 * np.cos(q1) * np.cos(q2) - L3 * np.cos(
            q1) * np.cos(q2) * np.cos(
            q5) + L4 * np.cos(q4) * np.sin(q1) * np.sin(q6) - L3 * np.sin(q1) * np.sin(q4) * np.sin(q5) + L3 * np.cos(
            q1) * np.cos(q4) * np.sin(
            q2) * np.sin(q5) + L4 * np.cos(q1) * np.sin(q2) * np.sin(q4) * np.sin(q6) - L4 * np.cos(q6) * np.sin(
            q1) * np.sin(q4) * np.sin(
            q5) - L4 * np.cos(q1) * np.cos(q2) * np.cos(q5) * np.cos(q6) + L4 * np.cos(q1) * np.cos(q4) * np.cos(
            q6) * np.sin(q2) * np.sin(q5)
        return [xtip, ytip, ztip]

    def fk_orientation(self, j1, j2, j3, j4, j5, j6):
        # R08
        r11 = np.cos(j1) * np.cos(j4) * np.cos(j6) - np.cos(j2) * np.cos(j5) * np.sin(j1) * np.sin(j6) - np.cos(
            j6) * np.sin(j1) * np.sin(j2) * np.sin(j4) + np.cos(j1) * np.sin(j4) * np.sin(j5) * np.sin(j6) + np.cos(
            j4) * np.sin(j1) * np.sin(j2) * np.sin(j5) * np.sin(j6)
        r12 = np.cos(j1) * np.cos(j5) * np.sin(j4) + np.cos(j2) * np.sin(j1) * np.sin(j5) + np.cos(j4) * np.cos(
            j5) * np.sin(j1) * np.sin(j2)
        r13 = np.cos(j1) * np.cos(j6) * np.sin(j4) * np.sin(j5) - np.cos(j2) * np.cos(j5) * np.cos(j6) * np.sin(
            j1) - np.cos(j1) * np.cos(j4) * np.sin(j6) + np.sin(j1) * np.sin(j2) * np.sin(j4) * np.sin(j6) + np.cos(
            j4) * np.cos(j6) * np.sin(j1) * np.sin(j2) * np.sin(j5)
        r21 = np.cos(j5) * np.sin(j2) * np.sin(j6) - np.cos(j2) * np.cos(j6) * np.sin(j4) + np.cos(j2) * np.cos(
            j4) * np.sin(j5) * np.sin(j6)
        r22 = np.cos(j2) * np.cos(j4) * np.cos(j5) - np.sin(j2) * np.sin(j5)
        r23 = np.cos(j5) * np.cos(j6) * np.sin(j2) + np.cos(j2) * np.sin(j4) * np.sin(j6) + np.cos(j2) * np.cos(
            j4) * np.cos(j6) * np.sin(j5)
        r31 = np.cos(j4) * np.cos(j6) * np.sin(j1) + np.cos(j1) * np.cos(j2) * np.cos(j5) * np.sin(j6) + np.cos(
            j1) * np.cos(j6) * np.sin(j2) * np.sin(j4) + np.sin(j1) * np.sin(j4) * np.sin(j5) * np.sin(j6) - np.cos(
            j1) * np.cos(j4) * np.sin(j2) * np.sin(j5) * np.sin(j6)
        r32 = np.cos(j5) * np.sin(j1) * np.sin(j4) - np.cos(j1) * np.cos(j2) * np.sin(j5) - np.cos(j1) * np.cos(
            j4) * np.cos(j5) * np.sin(j2)
        r33 = np.cos(j1) * np.cos(j2) * np.cos(j5) * np.cos(j6) - np.cos(j4) * np.sin(j1) * np.sin(j6) - np.cos(
            j1) * np.sin(j2) * np.sin(j4) * np.sin(j6) + np.cos(j6) * np.sin(j1) * np.sin(j4) * np.sin(j5) - np.cos(
            j1) * np.cos(j4) * np.cos(j6) * np.sin(j2) * np.sin(j5)

        # R06
        # r11 = np.cos(j2) * np.cos(j5) * np.sin(j1) * np.sin(j6) - np.cos(j1) * np.cos(j4) * np.cos(j6) + np.cos(
        #     j6) * np.sin(j1) * np.sin(j2) * np.sin(j4) - np.cos(j1) * np.sin(j4) * np.sin(j5) * np.sin(j6) - np.cos(
        #     j4) * np.sin(j1) * np.sin(j2) * np.sin(j5) * np.sin(j6)
        # r12 = np.cos(j1) * np.cos(j4) * np.sin(j6) + np.cos(j2) * np.cos(j5) * np.cos(j6) * np.sin(j1) - np.cos(
        #     j1) * np.cos(j6) * np.sin(j4) * np.sin(j5) - np.sin(j1) * np.sin(j2) * np.sin(j4) * np.sin(j6) - np.cos(
        #     j4) * np.cos(j6) * np.sin(j1) * np.sin(j2) * np.sin(j5)
        # r13 = - np.cos(j1) * np.cos(j5) * np.sin(j4) - np.cos(j2) * np.sin(j1) * np.sin(j5) - np.cos(j4) * np.cos(
        #     j5) * np.sin(j1) * np.sin(j2)
        # r21 = np.cos(j2) * np.cos(j6) * np.sin(j4) - np.cos(j5) * np.sin(j2) * np.sin(j6) - np.cos(j2) * np.cos(
        #     j4) * np.sin(j5) * np.sin(j6)
        # r22 = - np.cos(j5) * np.cos(j6) * np.sin(j2) - np.cos(j2) * np.sin(j4) * np.sin(j6) - np.cos(j2) * np.cos(
        #     j4) * np.cos(j6) * np.sin(j5)
        # r23 = np.sin(j2) * np.sin(j5) - np.cos(j2) * np.cos(j4) * np.cos(j5)
        # r31 = np.cos(j1) * np.cos(j4) * np.sin(j2) * np.sin(j5) * np.sin(j6) - np.cos(j1) * np.cos(j2) * np.cos(
        #     j5) * np.sin(j6) - np.cos(j1) * np.cos(j6) * np.sin(j2) * np.sin(j4) - np.sin(j1) * np.sin(j4) * np.sin(
        #     j5) * np.sin(j6) - np.cos(j4) * np.cos(j6) * np.sin(j1)
        # r32 = np.cos(j4) * np.sin(j1) * np.sin(j6) - np.cos(j1) * np.cos(j2) * np.cos(j5) * np.cos(j6) + np.cos(
        #     j1) * np.sin(j2) * np.sin(j4) * np.sin(j6) - np.cos(j6) * np.sin(j1) * np.sin(j4) * np.sin(j5) + np.cos(
        #     j1) * np.cos(j4) * np.cos(j6) * np.sin(j2) * np.sin(j5)
        # r33 = np.cos(j1) * np.cos(j2) * np.sin(j5) - np.cos(j5) * np.sin(j1) * np.sin(j4) + np.cos(j1) * np.cos(
        #     j4) * np.cos(j5) * np.sin(j2)

        # R_yaw_tooltip = np.array([[ 0.0, -1.0,  0.0], [ 0.0,  0.0,  1.0], [-1.0,  0.0,  0.0]])
        R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        # R.dot(R_yaw_tooltip)
        return R

    def ik_position(self, pos): # (m)
        x = pos[0]
        y = pos[1]
        z = pos[2]

        # Forward Kinematics
        # x = np.cos(q2)*np.sin(q1)*(L2-L1+q3)
        # y = -np.sin(q2)*(L2-L1+q3)
        # z = -np.cos(q1)*np.cos(q2)*(L2-L1+q3)

        # Inverse Kinematics
        q1 = np.arctan2(x, -z)     # (rad)
        q2 = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))  # (rad)
        q3 = np.sqrt(x ** 2 + y ** 2 + z ** 2) + self.L1 - self.L2  # (m)
        return q1, q2, q3

    def ik_orientation(self, q1,q2,Rb):
        R03 = np.array([[-np.sin(q1)*np.sin(q2), -np.cos(q1), np.cos(q2)*np.sin(q1)],
                        [-np.cos(q2), 0, -np.sin(q2)],
                        [np.cos(q1)*np.sin(q2), -np.sin(q1), -np.cos(q1)*np.cos(q2)]])
        R38 = R03.T.dot(Rb)
        r12 = R38[0,1]
        r22 = R38[1,1]
        r31 = R38[2,0]
        r32 = R38[2,1]
        r33 = R38[2,2]
        q4 = np.arctan2(-r22, -r12)     # (rad)
        q6 = np.arctan2(-r31, -r33)
        q5 = np.arctan2(r32, np.sqrt(r31**2+r33**2))
        return q4,q5,q6

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

            # Find balls and overlay
            pbs = BD.find_balls(img_color, img_depth, img_point)
            img_color = BD.overlay_balls(img_color, pbs)

            # Find tool position, joint angles, and overlay
            if pbs[0]==[] or pbs[1]==[]:
                pass
            else:
                # Find tool position, joint angles, and overlay
                pt = BD.find_tool_position(pbs[0], pbs[1])    # tool position of pitch axis
                pt = np.array(pt) * 0.001  # (m)
                pt = BD.Rrc.dot(pt) + BD.trc
                q1, q2, q3 = BD.ik_position(pt)
                # print(q1*180/np.pi, q2*180/np.pi, q3)
                img_color = BD.overlay_tool_position(img_color, [q1,q2,q3], (0,255,0))

                # Find tool orientation, joint angles, and overlay
                count_pbs = [pbs[2], pbs[3], pbs[4], pbs[5]]
                if count_pbs.count([]) >= 2:
                    pass
                else:
                    Rm = BD.find_tool_orientation(pbs[2],pbs[3],pbs[4],pbs[5])    # orientation of the marker
                    q4,q5,q6 = BD.ik_orientation(q1,q2,Rm)
                    # print(q4*180/np.pi,q5*180/np.pi,q6*180/np.pi)
                    print(q5*180/np.pi)
                    img_color = BD.overlay_tool(img_color, [q1, q2, q3, q4, q5, q6], (0,255,0))

            cv2.imwrite("ball_detected.png", img_color)
            cv2.imshow("images", img_color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        finally:
            pass