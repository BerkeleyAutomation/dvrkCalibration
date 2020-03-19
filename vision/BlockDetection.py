import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

class BlockDetection():
    def __init__(self):
        # data members
        self.mask_dx = 70  # mask size
        self.mask_dy = 70
        self.angles = np.r_[-60:60:1]    # angles for block rotation
        self.n_pegs = 12
        self.d_block = [858, 868]   # depth for block masking
        self.d_peg = [840, 857]     # depth for peg masking
        self.peg_pnts = []

        # load mask
        filename = '/home/hwangmh/pycharmprojects/FLSpegtransfer/img/block_mask_filled.png'
        self.mask = self.load_mask(filename, self.mask_dx, self.mask_dy, scaling=0.86)
        self.contour = self.load_contour(self.mask, 2)
        self.mask_points = [[p[1], p[0]] for p in np.argwhere(self.mask == 255)]    # w.r.t image coordinate

        # sample grasping points
        self.sample_gps = self.load_grasping_point(gp_number=3, dist_center=16, dist_gp=2)
        self.sample_pps = self.load_grasping_point(gp_number=3, dist_center=16, dist_gp=2)

        # camera intrinsic parameter
        self.__D = [-0.2826650142669678, 0.42553916573524475, -0.0005135679966770113, -0.000839113024994731,
                    -0.5215581655502319]
        self.__K = [[2772.32, 0.0, 956.3778], [0.0, 2772.50, 599.710], [0.0, 0.0, 1.0]]    # linear resgression
        self.__R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.__P = [[2776.604248046875, 0.0, 952.436279296875], [0.0, 0.0, 2776.226318359375],
                    [597.9248046875, 0.0, 0.0], [0.0,
                                                 1.0, 0.0]]
        self.__fx = self.__K[0][0]
        self.__fy = self.__K[1][1]
        self.__cx = self.__K[0][2]
        self.__cy = self.__K[1][2]

        # img cropping
        self.ycr = 430
        self.hcr = 400
        self.xcr = 680
        self.wcr = 520

    def img_crop(self, img_color, img_depth, img_point):
        color_cropped = img_color[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        depth_cropped = img_depth[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        point_cropped = img_point[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        return color_cropped, depth_cropped, point_cropped

    def pixel2world(self, x, y, depth):
        Xc = (x - self.__cx + self.xcr) / self.__fx * depth
        Yc = (y - self.__cy + self.ycr) / self.__fy * depth
        Zc = depth
        return Xc, Yc, Zc

    def world2pixel(self, Xc, Yc, Zc, Rc=0):
        x = self.__fx * Xc / Zc + self.__cx - self.xcr
        y = self.__fy * Yc / Zc + self.__cy - self.ycr
        r = (self.__fx+self.__fy)/2 * Rc / Zc
        return int(x), int(y), int(r)

    def load_mask(self, filename, size_x, size_y, scaling=1.0):
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

    def load_contour(self, mask, linewidth):
        dx = mask.shape[0]
        dy = mask.shape[1]
        edge = cv2.Canny(mask, dx, dy)
        contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = np.zeros((dx, dy), np.uint8)
        cv2.drawContours(img_contour, contours, -1, (255,255,255), linewidth)
        ret, img_contour = cv2.threshold(img_contour, 200, 255, cv2.THRESH_BINARY)
        return img_contour


    def load_grasping_point(self, gp_number, dist_center, dist_gp):
        group1 = []
        for i in range(gp_number):
            x = -(gp_number-1)*dist_gp//2
            y = dist_center
            group1.append([x+dist_gp*i+self.mask_dx//2, y+self.mask_dy//2])
        group0 = self.transform_pnts(group1, [self.mask_dx//2, self.mask_dy//2], -120, 0, 0)
        group2 = self.transform_pnts(group1, [self.mask_dx//2, self.mask_dy//2], 120, 0, 0)
        sample_grasping_point = np.reshape([group0, group1, group2], (gp_number*3,2))
        return sample_grasping_point


    def transform_img(self, img, rot_center, angle_deg, tx, ty):    # angle is positive in counter-clockwise
        R = cv2.getRotationMatrix2D((rot_center[0], rot_center[1]), angle_deg, 1)
        t = np.float32([[1, 0, tx], [0, 1, ty]])
        rotated = cv2.warpAffine(img, R, (img.shape[0], img.shape[1]))
        transformed = cv2.warpAffine(rotated, t, (img.shape[0], img.shape[1]))
        return transformed


    def transform_pnts(self, pnts, rot_center, angle_deg, tx, ty):
        pnts = np.array(pnts)
        R = cv2.getRotationMatrix2D((0,0), angle_deg, 1)[:,:2]
        t = np.array([tx, ty])
        new_pnts = [R.dot(p-rot_center) + t + rot_center for p in pnts]
        return new_pnts


    def pegs_detection(self, masked_img, n_pegs):
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
            peg_points = self.sort_position(peg_points)
        except:
            pass
        return peg_points

    def sort_position(self, points):
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

    def downsample_naive(self, img, downsample_factor):
        """
        Naively downsamples image without LPF.
        """
        new_img = img.copy()
        new_img = new_img[::downsample_factor]
        new_img = new_img[:, ::downsample_factor]
        return new_img


    def crop_blocks(self, blocks_masked):
        # crop each block around peg
        blocks_cropped = np.array([blocks_masked[c[1]-self.mask_dy//2:c[1] + self.mask_dy // 2,
                                   c[0]-self.mask_dx//2:c[0]+self.mask_dx//2] for c in self.peg_pnts])
        return blocks_cropped


    def find_blocks(self, blocks_cropped):
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
                mask_transformed = [self.transform_img(self.mask, (self.mask_dx//2, self.mask_dy//2), ang, tx, ty) for ang in self.angles]
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


    def find_blocks_global(self, pose_blks):
        # [block_numbering, angle, tx, ty, seen]
        pose_blks_global = [[res[0], res[1], self.peg_pnts[res[0]][0] + res[2] - self.mask_dx // 2,
                             self.peg_pnts[res[0]][1] + res[3] - self.mask_dy // 2, res[4]] for res in pose_blks]
        return pose_blks_global


    def find_blocks_depth(self, pose_blks_local, img_blks_masked_org, img_depth):
        # cropped images
        img_blks_org = self.crop_blocks(img_blks_masked_org)
        img_depth_crop = self.crop_blocks(img_depth)
        depth_blks = []
        for blk, d, res in zip(img_blks_org, img_depth_crop, pose_blks_local):
            n, ang, tx, ty, seen = res
            img_mask = self.transform_img(self.mask, (self.mask_dx//2, self.mask_dy//2), ang, tx, ty)   # mask
            img_blk = cv2.bitwise_and(blk, blk, mask=img_mask)
            arg = np.argwhere(img_blk==255)
            if len(arg) > 800:
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


    def sort_moving_order(self, pose_blks, order='l2r'):
        # pick & place action ordering
        action = np.array([[1,7], [3,8], [5,11], [0,6], [2,9], [4,10]])
        trial = [2,2,2,2,2,2]
        pose_blks_new = []
        if order=='r2l':
            action[:, [0, 1]] = action[:, [1, 0]]   # swapping the two columns
        for a,t in zip(action, trial):
            arg_pick = np.argwhere(np.array(pose_blks)[:, 0] == a[0])[0][0]
            arg_place = np.argwhere(np.array(pose_blks)[:, 0] == a[1])[0][0]
            if pose_blks[arg_pick][5] == True and pose_blks[arg_place][5] == False:    # if there is a block
                pose_blks_new.append(pose_blks[arg_pick])
                # Let placing pose be the same as the picking-up pose
                pose_blks[arg_place][1] = pose_blks[arg_pick][1]
                pose_blks_new.append(pose_blks[arg_place])
            else:
                pose_blks_new.append([])
                pose_blks_new.append([])
        return pose_blks_new    # [block_numbering, angle, tx, ty, depth, seen]


    def get_sample_grasping_pose(self, angle_rotated, x, y, which_sample='pick'):
        if which_sample=='pick':
            sample_points = self.sample_gps
        elif which_sample=='place':
            sample_points = self.sample_pps
        theta = angle_rotated
        gp_rotated = np.array(self.transform_pnts(sample_points, (self.mask_dx//2, self.mask_dy//2), theta, 0, 0))
        if theta > 0:
            grasping_angle_rotated = [-30 + theta, -30 + theta, -30 + theta,
                                      -90 + theta, -90 + theta, -90 + theta,
                                      30 + theta, 30 + theta, 30 + theta]
        else:
            grasping_angle_rotated = [-30 + theta, -30 + theta, -30 + theta,
                                      90 + theta, 90 + theta, 90 + theta,
                                      30 + theta, 30 + theta, 30 + theta]
            # [gp number, gp_angle, tx, ty]
        return [[i, ga, gp[0]+x, gp[1]+y] for i,(ga,gp) in enumerate(zip(grasping_angle_rotated, gp_rotated))]


    def find_all_grasping_pose(self, pose_blks):
        all_grasping_pose = []
        for res in pose_blks:
            n, theta, x, y, seen = res
            gp_rotated = np.array(self.get_sample_grasping_pose(theta, x, y, 'pick'))
            for gp in gp_rotated:
                all_grasping_pose.append([n, gp[0], gp[1], gp[2], gp[3], seen])
        return all_grasping_pose  # [peg number, gp number, theta, x, y, seen]


    # theta is positive in counter-clockwise direction
    def find_grasping_pose(self, pose_blks, which_side):
        grasping_pose = []
        pixel_coord = []
        for res in pose_blks:
            if res == []:
                grasping_pose.append([])
                pixel_coord.append([])
            else:
                n, ang, x, y, depth, seen = res
                if seen == True:    # pick up
                    # [gp number, gp_angle, tx, ty]
                    gp_rotated = np.array(self.get_sample_grasping_pose(ang, x, y, 'pick')).reshape(3,3,4)
                    gp_rotated_middle = gp_rotated[:,1,:]

                    # Choose the nearest point from the end effector
                    # a = 1.5  # slope of the line
                    if which_side == 'right_arm':
                        x_gp = gp_rotated_middle[:, 2]
                        arg = np.argsort(x_gp)[-2:]  # find maximum x
                        # k = gp_rotated_middle[:,3] - a * gp_rotated_middle[:,2]  # y = a*x + k
                        # arg = np.argmin(k)
                    elif which_side == 'left_arm':
                        x_gp = gp_rotated_middle[:, 2]
                        arg = np.argsort(x_gp)[:2]  # find minimum x
                        # k = gp_rotated_middle[:, 3] + a * gp_rotated_middle[:, 2]  # y = -a*x + k
                        # arg = np.argmin(k)

                    # Choose the farthest point from the peg among the three points on the same side
                    dist1 = np.linalg.norm(gp_rotated[arg[0]][:, 2:] - self.peg_pnts[n], axis=1)
                    dist2 = np.linalg.norm(gp_rotated[arg[1]][:, 2:] - self.peg_pnts[n], axis=1)
                    if max(dist1) > max(dist2):
                        arg = arg[0]
                        argmax = np.argmax(dist1)
                    else:
                        arg = arg[1]
                        argmax = np.argmax(dist2)
                    d = depth                   # depth should be the same in the placing pose
                else:
                    # [gp number, gp_angle, tx, ty]
                    gp_rotated = np.array(self.get_sample_grasping_pose(ang, x, y, 'place')).reshape(3, 3, 4)
                    # gp_rotated_middle = gp_rotated[:, 1, :]

                theta = gp_rotated[arg][argmax][1]      # grasping angle
                x_pixel = gp_rotated[arg][argmax][2]    # grasping position x
                y_pixel = gp_rotated[arg][argmax][3]    # grasping position y
                Xc, Yc, Zc = self.pixel2world(x_pixel, y_pixel, d)  # 3D position in respect to the camera (m)
                grasping_pose.append([n, theta, Xc, Yc, Zc, seen])
                pixel_coord.append([x_pixel, y_pixel])
        return grasping_pose, pixel_coord   # [peg number, theta, x, y, z, seen], [pixel_x, pixel_y]


    def overlay_blocks_and_pegs(self, img, peg_points):
        try:
            # Coloring yellow on blocks and pegs
            colored = self.change_color(img, (0, 255, 255))

            # Coloring pegs & put white letters numbering pegs
            count = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = [[-40,-20],[-40,-20],[-40,-20],[-40,-20],[-40,-20],[-40,-20],
                        [-20,-30],[-20,-30],[-20,-30],[-20,40],[-20,40],[-20,40]]
            for p,tp in zip(peg_points,text_pos):
                cv2.circle(colored, (p[0], p[1]), 5, (255, 0, 0), 2, -1)  # blue color overlayed on pegs
                text = "%d" % (count);
                cv2.putText(colored, text, (p[0]+tp[0], p[1]+tp[1]), font, 0.5, (255, 255, 255), 1)
                count += 1
        except:
            pass
        return colored


    def overlay_block_contour(self, img, pose_blks):
        overlayed = np.copy(img)
        # Overlay contour
        for res in pose_blks:
            if res==[]:
                pass
            else:
                n, theta, x, y, _, _ = res
                dx = self.contour.shape[1]
                dy = self.contour.shape[0]
                roi = overlayed[y:y + dy, x:x + dx]
                transformed = self.transform_img(self.contour, (self.mask_dx//2, self.mask_dy//2), theta, 0, 0)
                transformed_inv = cv2.bitwise_not(transformed)
                bg = cv2.bitwise_and(roi, roi, mask=transformed_inv)
                transformed_colored = self.change_color(transformed, (0, 255, 0))  # green color overlayed
                dst = cv2.add(bg, transformed_colored)
                overlayed[y:y + dy, x:x + dx] = dst
        return overlayed


    def overlay_pegs(self, img, peg_points, put_text=False):
        # Coloring white on blocks
        pegs_colored = self.change_color(img, (255, 255, 255))
        count = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for p in peg_points:
            cv2.circle(pegs_colored, (p[0], p[1]), 5, (0, 255, 0), 2, -1)   # green color overlayed
            if put_text:
                text = "%d" % (count);
                cv2.putText(pegs_colored, text, (p[0]-10, p[1]-10), font, 0.5, (255, 255, 255), 1)
                count += 1
        return pegs_colored


    def overlay_grasping_pose(self, img, grasping_pose, color):
        overlayed = np.copy(img)
        for gp in grasping_pose:
            if gp == []:
                pass
            else:
                gp = list(map(int, gp))
                cv2.circle(overlayed, (gp[0], gp[1]), 3, color, 2, -1)
        return overlayed


    def change_color(self, img, color):
        colored = np.copy(img)
        colored = cv2.cvtColor(colored, cv2.COLOR_GRAY2BGR)
        args = np.argwhere(colored)
        for n in args:
            colored[n[0]][n[1]] = list(color)
        return colored

    def peg_detection(self, img_depth, img_point):  # just for motion test
        # depth masking
        img_pegs_masked_org = cv2.inRange(img_depth, self.d_peg[0], self.d_peg[1])

        # noise filtering
        kernel = np.ones((2, 2), np.uint8)
        img_pegs_masked = cv2.morphologyEx(img_pegs_masked_org, cv2.MORPH_OPEN, kernel)

        # pegs detection: [[x0, y0], ..., [xn, yn]]
        self.peg_pnts = self.pegs_detection(img_pegs_masked, self.n_pegs)

        # crop pegs image
        img_crop_pegs = self.crop_blocks(img_pegs_masked_org)
        img_crop_points = self.crop_blocks(img_point)

        peg_points_3d = []
        for n, (peg, pt) in enumerate(zip(img_crop_pegs, img_crop_points)):
            arg = np.argwhere(peg == 255)
            points = np.array([pt[p[0], p[1]] for p in arg])
            height = np.nanmean(points[:, 2], axis=0) * 0.001  # (m)
            Xc, Yc, Zc = self.pixel2world(self.peg_pnts[n][0], self.peg_pnts[n][1], height)
            peg_points_3d.append([n, Xc, Yc, Zc])
        return peg_points_3d

    def FLSPerception(self, img_depth, img_point, order='l2r'):
        # depth masking
        img_blks_masked_org = cv2.inRange(img_depth, self.d_block[0], self.d_block[1])
        img_pegs_masked_org = cv2.inRange(img_depth, self.d_peg[0], self.d_peg[1])

        # noise filtering
        kernel = np.ones((2, 2), np.uint8)
        # img_blks_masked = cv2.morphologyEx(img_blks_masked_org, cv2.MORPH_OPEN, kernel)
        img_pegs_masked = cv2.morphologyEx(img_pegs_masked_org, cv2.MORPH_OPEN, kernel)

        # pegs detection: [[x0, y0], ..., [xn, yn]]
        self.peg_pnts = self.pegs_detection(img_pegs_masked, self.n_pegs)

        # Crop blocks around each peg
        img_blks_cr = self.crop_blocks(img_blks_masked_org)

        # Find pose of blocks from the segment images around peg_points
        # [block_number, angle, tx, ty, seen]
        pose_blks_local = self.find_blocks(img_blks_cr)
        pose_blks = self.find_blocks_global(pose_blks_local)

        # Find 3D points & depth of block
        depth_blks = self.find_blocks_depth(pose_blks_local, img_blks_masked_org, img_depth)   # [block number, depth] (mm)
        pose_blks = self.add_depth_to_result(pose_blks, depth_blks)         # [block number, angle, tx, ty, depth, seen]

        # Classify result of the blocks
        pose_blks = self.sort_moving_order(pose_blks, order=order)  # [block number, angle, tx, ty, depth, seen]

        # Find grasping pose: [block number, gp angle, Xc, Yc, Zc, seen]
        # all_gp = self.find_all_grasping_pose(pose_blks)
        gp, gp_pixel = self.find_grasping_pose(pose_blks, 'right_arm')

        # image overlay
        img_pegs_ovl = self.overlay_pegs(img_pegs_masked, self.peg_pnts, put_text=True)
        img_blks_ovl = cv2.add(img_blks_masked_org, img_pegs_masked)  # add circled pegs on image of blocks
        img_blks_ovl = self.overlay_blocks_and_pegs(img_blks_ovl, self.peg_pnts)  # put numbering and change color
        img_blks_ovl = self.overlay_block_contour(img_blks_ovl, pose_blks)  # draw contour along each block
        img_blks_ovl = self.overlay_grasping_pose(img_blks_ovl, gp_pixel, (0, 0, 255))
        # img_blks_ovl = self.overlay_grasping_pose(img_blks_ovl, all_gp, (255, 255, 255))
        return gp, img_pegs_ovl, img_blks_ovl

if __name__ == '__main__':
    # from FLSpegtransfer.vision.ZividCapture import ZividCapture
    BD = BlockDetection()
    img_color = []
    img_depth = []
    # zivid = ZividCapture()
    while True:
        image = np.load('../record/image.npy')
        depth = np.load('../record/depth.npy')
        point = np.load('../record/point.npy')
        img_color, img_depth, img_point = BD.img_crop(image, depth, point)
        # zivid.capture_3Dimage()
        # img_color, img_depth, img_point = BD.img_crop(zivid.image, zivid.depth, zivid.point)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
        if img_color == [] or img_depth == [] or img_point == []:
            pass
        else:
            # Find balls and overlay
            gp, img_pegs_ovl, img_blks_ovl = BD.FLSPerception(img_depth, img_point, order='l2r')
            cv2.imshow("color", img_color)
            cv2.imshow("depth", img_depth)
            cv2.imshow("pegs_overlayed", img_pegs_ovl)
            cv2.imshow("blocks_overlayed", img_blks_ovl)
            cv2.waitKey(1)