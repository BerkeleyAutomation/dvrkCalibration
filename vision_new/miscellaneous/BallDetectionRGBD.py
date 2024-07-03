import cv2
import numpy as np
from dvrk.motion.dvrkKinematics import dvrkKinematics
import dvrk.motion.dvrkVariables as dvrkVar
from dvrk.vision.cameras.ZividUtils import ZividUtils


class BallDetectionRGBD:
    # def __init__(self, Trc, Tpc, which_camera='inclined'):
    def __init__(self, Trc, which_camera="inclined"):

        # Transform
        # self.Tpc = Tpc  # from pegboard to camera
        # self.Tpc[:3, -1] *= 1000  # convert to (mm) as Zivid provides point sets in (mm)
        self.Trc = Trc  # from camera to robot
        if Trc == []:
            pass
        else:
            self.Rrc = self.Trc[:3, :3]
            self.trc = self.Trc[:3, 3]

        # thresholding value
        # self.__masking_depth = [-170, 20]
        if which_camera == "inclined":
            self.__masking_depth = [0, 1000]
        elif which_camera == "overhead":
            self.__masking_depth = [700, 2000]
        self.__lower_red = np.array([0 - 20, 100, 40])
        self.__upper_red = np.array([0 + 20, 255, 255])
        self.__lower_green = np.array([60 - 20, 160, 40])
        self.__upper_green = np.array([60 + 20, 255, 255])
        self.__lower_blue = np.array([120 - 30, 100, 40])
        self.__upper_blue = np.array([120 + 30, 255, 255])
        self.__lower_yellow = np.array([30 - 10, 130, 60])
        self.__upper_yellow = np.array([30 + 10, 255, 255])
        # radius of sphere fiducials = [12.0, 10.0, 8.0, 8.0, 8.0, 8.0]    # (mm)

        # dimension of tool
        self.d = 35  # length of coordinate (mm)
        # self.Lbb = 0.050  # ball1 ~ ball2 (m)
        # self.Lbp = 0.017  # ball2 ~ pitch (m)

        # new spheres
        self.Lbb = 0.03499  # ball1 ~ ball2 (m)
        self.Lbp = 0.01583  # ball2 ~ pitch (m)
        # D_ball1 = 20, D_ball2 = 17
        self.zivid_utils = ZividUtils(which_camera=which_camera)

    def img_crop(self, img_color, img_depth, img_point):
        color_cropped = img_color[self.__ycr : self.__ycr + self.__hcr, self.__xcr : self.__xcr + self.__wcr]
        depth_cropped = img_depth[self.__ycr : self.__ycr + self.__hcr, self.__xcr : self.__xcr + self.__wcr]
        point_cropped = img_point[self.__ycr : self.__ycr + self.__hcr, self.__xcr : self.__xcr + self.__wcr]
        return color_cropped, depth_cropped, point_cropped

    def pixel2world(self, x, y, depth):
        Xc = (x - self.zivid_utils.cx + self.zivid_utils.xcr) / self.zivid_utils.fx * depth
        Yc = (y - self.zivid_utils.cy + self.zivid_utils.ycr) / self.zivid_utils.fy * depth
        Zc = depth
        return Xc, Yc, Zc

    def world2pixel(self, Xc, Yc, Zc, Rc=0):
        x = self.zivid_utils.fx * Xc / Zc + self.zivid_utils.cx - self.zivid_utils.xcr
        y = self.zivid_utils.fy * Yc / Zc + self.zivid_utils.cy - self.zivid_utils.ycr
        r = (self.zivid_utils.fx + self.zivid_utils.fy) / 2 * Rc / Zc
        return int(x), int(y), int(r)

    def overlay_dot(self, img_color, pnt_3D, text):
        pnt = np.array(pnt_3D)
        if pnt.size == 0:
            return img_color
        else:
            overlayed = img_color.copy()
            img_pnt = self.world2pixel(pnt[0], pnt[1], pnt[2])
            cv2.putText(
                overlayed,
                text,
                (img_pnt[0] + 10, img_pnt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(overlayed, (img_pnt[0], img_pnt[1]), 3, (0, 255, 255), -1)
        return overlayed

    def overlay_ball(self, img_color, pbs):
        pbs = np.array(pbs)
        if pbs.size == 0:
            return img_color
        else:
            overlayed = img_color.copy()
            for i, pb in enumerate(pbs):
                pb_img = self.world2pixel(pb[0], pb[1], pb[2], pb[3])
                cv2.circle(overlayed, (pb_img[0], pb_img[1]), pb_img[2], (0, 255, 255), 2)
                cv2.putText(
                    overlayed,
                    str(i),
                    (pb_img[0] + 10, pb_img[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(overlayed, (pb_img[0], pb_img[1]), 3, (0, 255, 255), -1)
        return overlayed

    def overlay_tool(self, img_color, joint_angles, color):
        try:
            # 3D points w.r.t camera frame
            pb = self.Rrc.T.dot(np.array([0, 0, 0]) - self.trc) * 1000  # base position
            p5 = np.array(dvrkKinematics.fk(joint_angles, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0))[:3, 3]
            p5 = self.Rrc.T.dot(np.array(p5) - self.trc) * 1000  # pitch axis
            p6 = np.array(dvrkKinematics.fk(joint_angles, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=0))[:3, 3]
            p6 = self.Rrc.T.dot(np.array(p6) - self.trc) * 1000  # yaw axis
            p7 = np.array(
                dvrkKinematics.fk(joint_angles, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4 + 0.005)
            )[:3, 3]
            p7 = self.Rrc.T.dot(np.array(p7) - self.trc) * 1000  # tip

            pb_img = self.world2pixel(pb[0], pb[1], pb[2])
            p5_img = self.world2pixel(p5[0], p5[1], p5[2])
            p6_img = self.world2pixel(p6[0], p6[1], p6[2])
            p7_img = self.world2pixel(p7[0], p7[1], p7[2])

            overlayed = img_color.copy()
            self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0, 255, 0), 1, style="dotted", gap=8)
            cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
            self.drawline(overlayed, p5_img[0:2], p6_img[0:2], (0, 255, 0), 1, style="dotted", gap=8)
            cv2.circle(overlayed, p6_img[0:2], 2, color, 2)
            self.drawline(overlayed, p6_img[0:2], p7_img[0:2], (0, 255, 0), 1, style="dotted", gap=8)
            cv2.circle(overlayed, p7_img[0:2], 2, color, 2)
            return overlayed
        except:
            return img_color

    def overlay_tool_position(self, img_color, joint_angles, color):
        q1, q2, q3 = joint_angles
        # 3D points w.r.t camera frame
        pb = self.Rrc.T.dot(np.array([0, 0, 0]) - self.trc) * 1000  # base position
        p5 = np.array(dvrkKinematics.fk([q1, q2, q3, 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0))[:3, 3]
        p5 = self.Rrc.T.dot(np.array(p5) - self.trc) * 1000  # pitch axis

        pb_img = self.world2pixel(pb[0], pb[1], pb[2])
        p5_img = self.world2pixel(p5[0], p5[1], p5[2])

        overlayed = img_color.copy()
        self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0, 255, 0), 1, style="dotted", gap=8)
        cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
        return overlayed

    def drawline(self, img, pt1, pt2, color, thickness=1, style="dotted", gap=20):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
            pts.append((x, y))

        if style == "dotted":
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        elif style == "dashed":
            st = pts[0]
            ed = pts[0]
            i = 0
            for p in pts:
                st = ed
                ed = p
                if i % 2 == 1:
                    cv2.line(img, st, ed, color, thickness)
                i += 1

    @classmethod
    def transform(cls, points, T):
        R = T[:3, :3]
        t = T[:3, -1]
        return R.dot(points.T).T + t.T

    def mask_image(self, img_color, img_point, color, visualize=False):
        try:
            # define hsv_range
            if color == "red":
                hsv_range = [self.__lower_red, self.__upper_red]
            elif color == "green":
                hsv_range = [self.__lower_green, self.__upper_green]
            elif color == "blue":
                hsv_range = [self.__lower_blue, self.__upper_blue]
            elif color == "yellow":
                hsv_range = [self.__lower_yellow, self.__upper_yellow]
            else:
                hsv_range = []

            # 2D color masking
            img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
            img_masked = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])

            if visualize:
                cv2.imshow("", img_masked)
                cv2.waitKey(0)

            # noise filtering
            # kernel = np.ones((2, 2), np.uint8)
            # img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel)

            # color masking
            con1 = img_masked == 255
            arg1 = np.argwhere(con1)
            pnt1 = img_point[con1]

            if len(arg1) < 1000:
                mask = np.zeros_like(img_masked)
            else:
                # remove nan
                con2 = ~np.isnan(pnt1).any(axis=1)
                arg2 = np.argwhere(con2)
                pnt2 = pnt1[con2]

                # transform w.r.t. task coordinate
                # pnt2_tr = self.transform(pnt2, self.Tpc)

                # depth masking
                con3 = (pnt2[:, 2] > self.__masking_depth[0]) & (pnt2[:, 2] < self.__masking_depth[1])
                arg3 = np.argwhere(con3)

                # creat mask where color & depth conditions hold
                arg_mask = np.squeeze(arg1[arg2[arg3]])
                mask = np.zeros_like(img_masked)
                mask[arg_mask[:, 0], arg_mask[:, 1]] = 255
            return mask
            # return img_masked
        except:
            return []

    def find_balls(self, img_color, img_point, color, nb_sphere, visualize=False):
        try:
            if visualize:
                cv2.imshow("", img_color)
                cv2.waitKey(0)

            # mask color & depth
            masked = self.mask_image(img_color, img_point, color, visualize)
            if visualize:
                cv2.imshow("", masked)
                cv2.waitKey(0)

            # Find contours
            cnts, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:nb_sphere]
            pb = []
            for c in cnts:
                if len(c) < 50:
                    # if cv2.contourArea(c) > 500: # thresholding by area is more accurate
                    pass
                else:
                    # if visualize:
                    img_color_copy = np.copy(img_color)
                    cv2.drawContours(img_color_copy, [c], -1, (0, 255, 255), 1)
                    if visualize:
                        cv2.imshow("", img_color_copy)
                        cv2.waitKey(0)

                    # Find 3D points of a ball
                    # Get the pixel coordinates inside the contour
                    infilled = np.zeros(np.shape(img_color), np.uint8)
                    cv2.drawContours(infilled, [c], 0, (255, 255, 255), -1)
                    infilled = cv2.cvtColor(infilled, cv2.COLOR_BGR2GRAY)
                    ball_masked = cv2.bitwise_and(masked, masked, mask=infilled)
                    if visualize:
                        cv2.imshow("", ball_masked)
                        cv2.waitKey(0)

                    # Get the point clouds
                    args = np.argwhere(ball_masked == 255)
                    points_ball = img_point[args[:, 0], args[:, 1]]

                    # Linear regression to fit the circle into the point cloud
                    xc, yc, zc, rc = self.fit_circle_3d(points_ball[:, 0], points_ball[:, 1], points_ball[:, 2])
                    pb.append([xc, yc, zc, rc])

            if len(pb) < nb_sphere:
                return []

            if len(pb) >= 2:
                # sort by radius
                pb = np.array(pb)
                arg = np.argsort(pb[:, 3])[::-1]
                return pb[arg]
            else:
                return np.squeeze(pb)
        except:
            return []

    @classmethod
    def fit_circle_3d(cls, x, y, z, w=[]):
        A = np.array([x, y, z, np.ones(len(x))]).T
        b = x**2 + y**2 + z**2

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
        r = np.sqrt(c[3] + xc**2 + yc**2 + zc**2)
        return xc, yc, zc, r

    # Get tool position of the pitch axis from two ball positions w.r.t. camera base coordinate
    def find_tool_pitch(self, pb1, pb2):
        pb1 = np.asarray(pb1[0:3], dtype=float)
        pb2 = np.asarray(pb2[0:3], dtype=float)
        # lbb=np.linalg.norm(pb1-pb2)
        # print("lbb",lbb,self.Lbb)
        # p_pitch = ((lbb+self.Lbp)*pb2-self.Lbp*pb1)/lbb
        p_pitch = ((self.Lbb + self.Lbp) * pb2 - self.Lbp * pb1) / self.Lbb
        return p_pitch  # (mm), w.r.t. camera base coordinate

    # Get orientation from three ball positions w.r.t. robot base coordinate
    def find_tool_orientation(self, pbr, pbg, pbb, pby, which_arm):
        pbr = np.array(pbr)  # red
        pbg = np.array(pbg)  # green
        pbb = np.array(pbb)  # blue
        pby = np.array(pby)  # yellow

        pts1 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])  # base pose w.r.t. robot coordinate

        # normalized direction vectors
        if which_arm == "PSM1":
            if pbr.size == 0:  # red ball is occluded
                pborg = (pbg + pby) / 2
                v1 = self.Rrc.dot((pborg - pbb)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pbg - pby)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif pbg.size == 0:  # green ball is occluded
                pborg = (pbr + pbb) / 2
                v1 = self.Rrc.dot((pbr - pbb)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pborg - pby)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif pbb.size == 0:  # blue ball is occluded
                pborg = (pbg + pby) / 2
                v1 = self.Rrc.dot((pbr - pborg)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pbg - pby)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif pby.size == 0:  # yellow ball is occluded
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
        elif which_arm == "PSM2":
            if pbr.size == 0:  # red ball is occluded
                pborg = (pbg + pby) / 2
                v1 = self.Rrc.dot((pbb - pborg)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pby - pbg)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif pbg.size == 0:  # green ball is occluded
                pborg = (pbr + pbb) / 2
                v1 = self.Rrc.dot((pbb - pbr)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pby - pborg)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif pbb.size == 0:  # blue ball is occluded
                pborg = (pbg + pby) / 2
                v1 = self.Rrc.dot((pborg - pbr)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pby - pbg)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif pby.size == 0:  # yellow ball is occluded
                pborg = (pbr + pbb) / 2
                v1 = self.Rrc.dot((pbb - pbr)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pborg - pbg)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            else:
                pborg = (pbg + pby) / 2
                v1 = self.Rrc.dot((pbb - pborg)[0:3])
                v1 = v1 / np.linalg.norm(v1)
                v2 = self.Rrc.dot((pby - pbg)[0:3])
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        else:
            raise ValueError
        pts2 = np.array([v1, v2, v3])
        pts1 = pts1.T
        pts2 = pts2.T
        Rb = pts2.dot(np.linalg.inv(pts1))
        return Rb  # w.r.t. robot base coordinate
