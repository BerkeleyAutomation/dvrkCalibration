import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from dvrk.utils.ImgUtils import ImgUtils


class BlockDetection2D:
    def __init__(self):
        # data members
        self.angles = np.r_[-60:60:1]    # angles for block rotation
        self.n_pegs = 12
        self.depth_block = [858, 868]   # depth for block masking
        self.depth_peg = [840, 857]     # depth for peg masking
        self.lower_red = np.array([0 - 20, 60, 40])    # color range for masking
        self.upper_red = np.array([0 + 20, 255, 255])
        # self.peg_pnts = []

        # load mask
        self.mask_dx = 70  # mask size
        self.mask_dy = 70
        filename = '/home/hwangmh/pycharmprojects/FLSpegtransfer/img/block_mask_filled.png'
        self.mask = self.load_mask(filename, self.mask_dx, self.mask_dy, scaling=0.86)

    # load a reference mask of block
    @classmethod
    def load_mask(cls, filename, size_x, size_y, scaling=1.0):
        # load mask
        mask_inv = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ret, mask_inv = cv2.threshold(mask_inv, 50, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask_inv)

        # scaling
        mask_fitted = cv2.resize(mask, dsize=(size_y, size_x))  # fit to the specified size
        h = int(size_y * scaling)
        w = int(size_x * scaling)
        mask_scaled = cv2.resize(mask_fitted, dsize=(h, w))

        # fit to the given size
        dh = abs(size_y - h)
        dw = abs(size_x - w)
        if h > size_y:
            mask_sample = mask_scaled[dw:size_y, dh:size_x]
        else:
            mask_sample = np.zeros((size_y, size_x), np.uint8)
            mask_sample[dw//2:dw//2+w, dh//2:dh//2+h] = mask_scaled
        return mask_sample

    @classmethod
    def mask_image(cls, img_color, img_depth, depth_range, hsv_range, noise_filtering=False):
        # depth masking
        depth_mask = cv2.inRange(img_depth, depth_range[0], depth_range[1])

        # color masking
        color_mask = cv2.bitwise_and(img_color, img_color, mask=depth_mask)
        img_hsv = cv2.cvtColor(color_mask, cv2.COLOR_BGR2HSV)
        img_masked = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])

        # noise filtering
        if noise_filtering:
            kernel = np.ones((2, 2), np.uint8)
            img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel)
        return img_masked

    @classmethod
    def sort_pegs(cls, points):
        points = np.array(points)
        arg_x = np.argsort(points[:, 0])
        g1 = points[arg_x][:3]
        g2 = points[arg_x][3:6]
        g3 = points[arg_x][6:8]
        g4 = points[arg_x][8:10]
        g5 = points[arg_x][10:12]

        g1 = g1[np.argsort(g1[:,1])]
        g2 = g2[np.argsort(g2[:,1])]
        g3 = g3[np.argsort(g3[:,1])]
        g4 = g4[np.argsort(g4[:,1])]
        g5 = g5[np.argsort(g5[:,1])]
        p1=g1[0]; p2=g2[0]; p3=g1[1]; p4=g2[1]; p5=g1[2]; p6=g2[2]
        p7=g3[0]; p8=g4[0]; p9=g5[0]; p10=g3[1]; p11=g4[1]; p12=g5[1]
        sorted = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
        return sorted

    @classmethod
    def crop_blocks(cls, blocks_masked, peg_pnts, dx, dy):
        # crop each block around peg
        blocks_cropped = np.array([blocks_masked[c[1]-dy//2:c[1]+dy//2, c[0]-dx//2:c[0]+dx//2] for c in peg_pnts])
        return blocks_cropped

    @classmethod
    def find_pegs(cls, masked_img, n_pegs):
        peg_points = []
        try:
            corners = cv2.goodFeaturesToTrack(masked_img, n_pegs, 0.1, 30)  # corner detection
            corners = np.int0(corners)
            corners = np.array([i.ravel() for i in corners])  # positions of pegs

            args = np.argwhere(masked_img > 10)
            dx = 20
            dy = 20
            peg_points = []
            for p in corners:
                args_y = np.argwhere((p[1]-dy<args[:,0]) & (args[:,0]<p[1]+dy))
                args_x = np.argwhere((p[0]-dx<args[:,1]) & (args[:,1]<p[0]+dx))
                common = np.intersect1d(args_x, args_y)
                average = np.average(args[common], axis=0)
                peg_points.append([average[1], average[0]])

            peg_points = np.array(peg_points).astype(int)
        except:
            pass
        return peg_points

    def match_blocks(self, blocks_cropped):
        pose_blks = []
        for n, b in enumerate(blocks_cropped):
            cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            if len(cnts)>=1 and cv2.contourArea(cnts[0]) > 400:
                # get center of triangle
                mmt = cv2.moments(cnts[0])
                cx = int(mmt['m10'] / mmt['m00'])
                cy = int(mmt['m01'] / mmt['m00'])
                tx = cx - self.mask_dx//2
                ty = cy - self.mask_dy//2

                # get clean and filled block image
                block = np.zeros_like(b)
                cv2.drawContours(block, cnts, 0, (255,255,255), -1)
                mask_transformed = [ImgUtils.transform_img(self.mask, (self.mask_dx//2, self.mask_dy//2), ang, tx, ty) for ang in self.angles]
                n_cross = np.zeros_like(self.angles)
                for i,mask in enumerate(mask_transformed):
                    block_crossed = cv2.bitwise_and(block, block, mask=mask)
                    n_cross[i] = np.shape(np.argwhere(block_crossed == 255))[0]

                # best matched angle
                best = np.argmax(n_cross)
                angle = self.angles[best]
                pose_blks.append([n, angle, tx, ty, True])  # [block_numbering, angle, tx, ty, seen]
            else:
                pose_blks.append([n, -30, 0, 0, False])
        return pose_blks    # pose_block

    def match_blocks_global(self, pose_blks, peg_pnts):
        # [block_numbering, angle, tx, ty, seen]
        pose_blks_global = [[res[0], res[1], peg_pnts[res[0]][0] + res[2] - self.mask_dx // 2,
                             peg_pnts[res[0]][1] + res[3] - self.mask_dy // 2, res[4]] for res in pose_blks]
        return pose_blks_global

    def find_blocks_depth(self, pose_blks_local, img_blks_masked_org, img_depth, peg_pnts):
        # cropped images
        img_blks_org = self.crop_blocks(img_blks_masked_org, peg_pnts, self.mask_dx, self.mask_dy)
        img_depth_crop = self.crop_blocks(img_depth, peg_pnts, self.mask_dx, self.mask_dy)
        depth_blks = []
        for blk, d, res in zip(img_blks_org, img_depth_crop, pose_blks_local):
            n, ang, tx, ty, seen = res
            img_mask = ImgUtils.transform_img(self.mask, (self.mask_dx//2, self.mask_dy//2), ang, tx, ty)   # mask
            img_blk = cv2.bitwise_and(blk, blk, mask=img_mask)
            arg = np.argwhere(img_blk==255)
            if len(arg) > 600:
                pts_depth = np.array([d[p[0], p[1]] for p in arg])
                depth_blk_avg = np.nanmean(pts_depth, axis=0) * 0.001  # (m)
                depth_blks.append([n, depth_blk_avg[0]])
            else:
                depth_blks.append([n, 0])
        return depth_blks


    def add_depth_to_result(self, pose_blks, depths):  # [block number, angle, tx, ty, depth, seen]
        pose_blks_new = []
        for p, depth in zip(pose_blks, depths):
            n, ang, tx, ty, seen = p
            _, d = depth
            pose_blks_new.append([n, ang, tx, ty, d, seen])
        return pose_blks_new


    def find_blocks(self, img_color, img_depth):
        img_blks = self.mask_image(img_color, img_depth, self.depth_block, [self.lower_red, self.upper_red])
        img_pegs = self.mask_image(img_color, img_depth, self.depth_peg, [self.lower_red, self.upper_red])

        # pegs detection: [[x0, y0], ..., [xn, yn]]
        peg_pnts = self.find_pegs(img_pegs, self.n_pegs)
        peg_pnts = self.sort_pegs(peg_pnts)

        # Crop blocks around each peg
        img_blks_cr = self.crop_blocks(img_blks, peg_pnts, self.mask_dx, self.mask_dy)

        # Find pose of blocks from the segment images around peg_points
        # [block_number, angle, tx, ty, seen]
        pose_blks_local = self.match_blocks(img_blks_cr)
        pose_blks = self.match_blocks_global(pose_blks_local, peg_pnts)

        # Find 3D points & depth of block
        depth_blks = self.find_blocks_depth(pose_blks_local, img_blks, img_depth, peg_pnts)   # [block number, depth] (mm)
        pose_blks = self.add_depth_to_result(pose_blks, depth_blks)         # [block number, angle, tx, ty, depth, seen]
        return pose_blks, img_blks, img_pegs, peg_pnts