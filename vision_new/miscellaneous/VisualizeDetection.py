import cv2
import numpy as np
from dvrk.utils.ImgUtils import ImgUtils
from mayavi import mlab

class VisualizeDetection():
    def __init__(self, BlockDetection=[]):
        if BlockDetection != []:
            self.mask = BlockDetection.mask
            self.mask_dx = BlockDetection.mask_dx
            self.mask_dy = BlockDetection.mask_dy
            self.contour = self.load_contour(self.mask, linewidth=2)

    @classmethod
    def load_contour(cls, mask, linewidth):
        dx = mask.shape[0]
        dy = mask.shape[1]
        edge = cv2.Canny(mask, dx, dy)
        contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = np.zeros((dx, dy), np.uint8)
        cv2.drawContours(img_contour, contours, -1, (255,255,255), linewidth)
        ret, img_contour = cv2.threshold(img_contour, 200, 255, cv2.THRESH_BINARY)
        return img_contour

    def change_color(self, img, color):
        assert np.ndim(img) == 2
        colored = np.copy(img)
        colored = cv2.cvtColor(colored, cv2.COLOR_GRAY2BGR)
        args = np.argwhere(colored)
        for n in args:
            colored[n[0]][n[1]] = list(color)
        return colored

    def overlay_pegs(self, pegs_colored, peg_points, put_text=False):
        count = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for p in peg_points:
            cv2.circle(pegs_colored, (p[0], p[1]), 5, (255, 0, 0), 2, -1)   # blue color overlayed
            if put_text:
                text = "%d" % (count);
                cv2.putText(pegs_colored, text, (p[0]-10, p[1]-10), font, 0.5, (255, 255, 255), 1)
                count += 1
        return pegs_colored

    # overlay contour
    def overlay_block(self, img, pose_blks):
        assert np.ndim(img) == 3
        overlayed = np.copy(img)
        for res in pose_blks:
            if res==[]:
                pass
            else:
                n, theta, x, y, _, _ = res
                dx = self.contour.shape[1]
                dy = self.contour.shape[0]
                roi = overlayed[y:y + dy, x:x + dx]
                transformed = ImgUtils.transform_img(self.contour, (self.mask_dx//2, self.mask_dy//2), theta, 0, 0)
                transformed_inv = cv2.bitwise_not(transformed)
                bg = cv2.bitwise_and(roi, roi, mask=transformed_inv)
                transformed_colored = self.change_color(transformed, (0, 255, 0))  # green color overlayed
                dst = cv2.add(bg, transformed_colored)
                overlayed[y:y + dy, x:x + dx] = dst
        return overlayed

    def overlay_grasping_pose(self, img, grasping_pose, color):
        assert np.ndim(img) == 3
        overlayed = np.copy(img)
        for gp in grasping_pose:
            if gp == []:
                pass
            else:
                gp = list(map(int, gp))
                cv2.circle(overlayed, (gp[2], gp[3]), 3, color, 2, -1)
        return overlayed

    def overlay(self, img_blks, img_pegs, pose_blks, peg_points):
        img_blks_pegs = cv2.add(img_blks, img_pegs)
        img_colored = self.change_color(img_blks_pegs, [0, 255, 255])  # color to yellow
        img_overlayed = self.overlay_pegs(img_colored, peg_points, put_text=False)
        img_overlayed = self.overlay_block(img_overlayed, pose_blks)
        return img_overlayed

    def plot3d(self, pnt_blocks=[], pnt_masks=[], pnt_pegs=[], pnt_grasping1=[], pnt_grasping2=[]):
        pnt_blocks = np.array(pnt_blocks)
        pnt_masks = np.array(pnt_masks)
        pnt_pegs = np.array(pnt_pegs)
        pnt_grasping1 = np.array(pnt_grasping1)
        pnt_grasping2 = np.array(pnt_grasping2)
        mlab.figure("FLS Peg Transfer", fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(1200, 900))  # black background
        for pnt_mask in pnt_masks:
            if pnt_mask != []:
                mlab.points3d(pnt_mask[:, 0], pnt_mask[:, 1], pnt_mask[:, 2], color=(0.0, 1.0, 0.0), scale_factor=0.3)  # green on masks
        for pnt_block in pnt_blocks:
            if pnt_block != []:
                mlab.points3d(pnt_block[:, 0], pnt_block[:, 1], pnt_block[:, 2], color=(1.0, 1.0, 0.0), scale_factor=.5)    # yellow on blocks
        if pnt_grasping1 != []:
            mlab.points3d(pnt_grasping1[:, 0], pnt_grasping1[:, 1], pnt_grasping1[:, 2], color=(1.0, 0.0, 0.0), scale_factor=1.7)  # red on grasping point
        if pnt_grasping2 != []:
            mlab.points3d(pnt_grasping2[:, 0], pnt_grasping2[:, 1], pnt_grasping2[:, 2], color=(0.0, 0.0, 1.0), scale_factor=1.7)  # blue on grasping point
        if pnt_pegs != []:
            mlab.points3d(pnt_pegs[:, 0], pnt_pegs[:, 1], pnt_pegs[:, 2], color=(0.0, 0.0, 0.0), scale_factor=3.2)  # black on pegs
        # mlab.axes(xlabel='x', ylabel='y', zlabel='z', z_axis_visibility=False)
        mlab.orientation_axes()
        # mlab.outline(color=(.7, .7, .7))
        mlab.view(azimuth=180, elevation=180)
        mlab.show()