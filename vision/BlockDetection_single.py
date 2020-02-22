import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import imutils

class BlockDetection():
    def __init__(self):
        # data members
        self.__mask = []
        self.__mask_transformed = []
        self.__contour = []
        self.mask_dx = 70  # mask size
        self.mask_dy = 70
        self.__theta = np.r_[-60:60:4]
        self.__x = np.r_[-12:12:4]
        self.__y = np.r_[-12:12:4]
        self.__number_pegs = 12
        self.__masking_depth_block = [872, 884]
        self.__masking_depth_peg = [855, 870]

        # load mask
        filename = '/home/hwangmh/pycharmprojects/FLSpegtransfer/img/block_mask.png'
        self.__mask = self.load_mask(filename, self.mask_dx, self.mask_dy, scaling=1.0)
        self.__mask_transformed = self.transform_mask(self.__mask)
        self.__contour = self.load_contour(self.__mask, 2)

        # sample grasping points
        self.__sample_grasping_points = self.load_grasping_point(gp_number=3, dist_center=23, dist_gp=3)
        self.__sample_placing_points = self.load_grasping_point(gp_number=3, dist_center=12, dist_gp=3)

        self.__ycr = 430
        self.__hcr = 400
        self.__xcr = 680
        self.__wcr = 520

    def img_crop(self, img_color, img_depth, img_point):
        color_cropped = img_color[self.__ycr:self.__ycr + self.__hcr, self.__xcr:self.__xcr + self.__wcr]
        depth_cropped = img_depth[self.__ycr:self.__ycr + self.__hcr, self.__xcr:self.__xcr + self.__wcr]
        point_cropped = img_point[self.__ycr:self.__ycr + self.__hcr, self.__xcr:self.__xcr + self.__wcr]
        return color_cropped, depth_cropped, point_cropped

    def load_intrinsics(self, filename):
        # load calibration data
        with np.load(filename) as X:
            _, mtx, dist, _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
        return mtx, dist

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
            group1.append([x+dist_gp*i+self.__mask.shape[1]//2, y+self.__mask.shape[0]//2])

        group0 = self.transform_pnt(group1, self.__mask.shape, -120, 0, 0)
        group2 = self.transform_pnt(group1, self.__mask.shape, 120, 0, 0)
        sample_grasping_point = np.reshape([group0, group1, group2], (gp_number*3,2))
        return sample_grasping_point

    # def rotate_mask(self, theta):
    #     rotated = np.zeros((len(self.__theta), len(self.__x), len(self.__y), np.shape(mask)[0], np.shape(mask)[1]),
    #                            np.uint8)
    #     for n, ang in enumerate(self.__theta):
    #         for i, tx in enumerate(self.__x):
    #             for j, ty in enumerate(self.__y):
    #                 transformed[n][i][j] = self.transform_img(mask, ang, tx, ty)
    #     return transformed

    def transform_mask(self, mask):
        transformed = np.zeros((len(self.__theta),len(self.__x),len(self.__y),np.shape(mask)[0],np.shape(mask)[1]), np.uint8)
        for n, ang in enumerate(self.__theta):
            for i, tx in enumerate(self.__x):
                for j, ty in enumerate(self.__y):
                    transformed[n][i][j] = self.transform_img(mask, ang, tx, ty)
        return transformed

    def transform_img(self, img, angle_deg, tx, ty):
        """
        :param img:
        :param angle_deg: positive in counter clockwise
        :param tx: positive in rightward
        :param ty: positive in downward
        :return:
        """
        M_rot = cv2.getRotationMatrix2D((img.shape[0]//2, img.shape[1]//2), angle_deg, 1)
        M_tran = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M_rot, (img.shape[0], img.shape[1]))
        rotated = cv2.warpAffine(img, M_tran, (img.shape[0], img.shape[1]))
        return rotated

    def transform_pnt(self, pnts, size_img, angle_deg, tx, ty):
        shifted = [[p[0]-size_img[0]//2, p[1]-size_img[1]//2] for p in pnts]
        R = cv2.getRotationMatrix2D((0,0), angle_deg, 1)[:,:2]
        T = np.array([tx, ty])
        transformed = [np.array(np.matmul(R, p) + T) for p in shifted]
        new_pnts = [[p[0]+size_img[0]//2, p[1]+size_img[1]//2] for p in transformed]
        return new_pnts

    def pegs_detection(self, masked_img, number_of_pegs):
        peg_points = []
        try:
            corners = cv2.goodFeaturesToTrack(masked_img, number_of_pegs, 0.1, 30)  # corner detection
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

    def rotate_mask(self, angle):
        mask = np.load("mask.npy")
        rotated = imutils.rotate_bound(mask, angle)
        rotated[rotated > 0] = 1
        return rotated

    def find_blocks2(self, blocks_masked, peg_points):
        # Segmenting block around each peg
        # block = np.array([blocks_masked[c[1] - self.mask_dy//2:c[1]+self.mask_dy//2, c[0]-self.mask_dx//2:c[0] + self.mask_dx // 2] for c in peg_points])
        # for n, b in enumerate(block):
        pass

    def find_blocks(self, blocks_masked, peg_points):
        try:
            result_blocks = []
            result_wo_blocks = []
            result_global_wo_blocks = []
            result_global_blocks = []
            # Segmenting block around each peg
            block = np.array(
                [blocks_masked[c[1] - self.mask_dy // 2:c[1] + self.mask_dy // 2, c[0] - self.mask_dx // 2:c[0] + self.mask_dx // 2] for c
                 in peg_points])
            n_criteria = 250    # number of points that overlapped with the sample mask

            for n, b in enumerate(block):
                n_max = 0
                for k, ang in enumerate(self.__theta):
                    for i, tx in enumerate(self.__x):
                        for j, ty in enumerate(self.__y):
                            block_crossed = cv2.bitwise_and(b, b, mask=self.__mask_transformed[k][i][j])
                            n_cross = np.shape(np.argwhere(block_crossed == 255))[0]
                            if n_cross > n_max:
                                n_max = n_cross
                                theta_final = ang
                                x_final = tx
                                y_final = ty
                if n_max > n_criteria:  # if there is a block,
                    result_blocks.append([n+1, theta_final, x_final, y_final])  # n is the numbering of blocks
                else:
                    result_wo_blocks.append([n+1, -30, 0, 0])

            result_global_blocks = [[res[0], res[1], peg_points[res[0]-1][0]+res[2]-self.__mask.shape[0]//2, peg_points[res[0]-1][1]+res[3]-self.__mask.shape[1]//2] for res in result_blocks]
            result_global_wo_blocks = [[res[0], res[1], peg_points[res[0]-1][0]+res[2]-self.__mask.shape[0]//2, peg_points[res[0]-1][1]+res[3]-self.__mask.shape[1]//2] for res in result_wo_blocks]
        except:
            pass
        return result_global_blocks, result_global_wo_blocks

    def classify_blocks(self, result_global_blocks, result_global_wo_blocks):
        try:
            pair_rarm = [[2,9], [4,12], [6,11], [1,8], [3,7], [5,10], [8,1], [9,2], [12, 4], [7,3], [10,5], [11,6]]
            pickup_blocks_rarm = []
            place_blocks_rarm = []
            for pr in pair_rarm:
                pick_number = pr[0]
                place_number = pr[1]
                # Find a row corresponding to the picking number
                arg_pick = np.argwhere(np.array(result_global_blocks)[:, 0] == pick_number)
                arg_place = np.argwhere(np.array(result_global_wo_blocks)[:, 0] == place_number)
                if len(arg_pick)==0 or len(arg_place)==0:  # empty
                    pass
                else:
                    arg_pick = arg_pick[0][0]
                    arg_place = arg_place[0][0]
                    pickup_blocks_rarm.append(result_global_blocks[arg_pick])
                    # result_global_wo_blocks[arg_place][1] = result_global_blocks[arg_pick][1]
                    place_blocks_rarm.append(result_global_wo_blocks[arg_place])
        except:
            pass
        return pickup_blocks_rarm, place_blocks_rarm

    def get_sample_grasping_pose(self, angle_rotated, x, y, which_sample='pick'):
        if which_sample=='pick':
            sample_points = self.__sample_grasping_points
        elif which_sample=='place':
            sample_points = self.__sample_placing_points
        theta = angle_rotated
        gp_rotated = np.array(self.transform_pnt(sample_points, self.__mask.shape, theta, 0, 0))

        if theta > 0:
            grasping_angle_rotated = [-30 + theta, -30 + theta, -30 + theta,
                                      -90 + theta, -90 + theta, -90 + theta,
                                      30 + theta, 30 + theta, 30 + theta]
        else:
            grasping_angle_rotated = [-30 + theta, -30 + theta, -30 + theta,
                                      90 + theta, 90 + theta, 90 + theta,
                                      30 + theta, 30 + theta, 30 + theta]

        return [[i, ga, gp[0]+x, gp[1]+y] for i,(ga,gp) in enumerate(zip(grasping_angle_rotated, gp_rotated))]

    def find_all_grasping_pose(self, result_global):
        all_grasping_pose = []
        for res in result_global:
            n, theta, x, y = res
            gp_rotated_global = np.array(self.get_sample_grasping_pose(theta, x, y, 'pick'))
            for gp in gp_rotated_global:
                all_grasping_pose.append([n, gp[0], gp[1], gp[2], gp[3]])
        return all_grasping_pose  # [peg number, gp number, theta, x, y]

    # theta is positive in counter-clockwise direction
    def find_grasping_pose(self, result_pickup_block, peg_points, which_side):
        final_grasping_pose = []
        for res in result_pickup_block:
            n, theta, x, y = res
            gp_rotated_global = np.array(self.get_sample_grasping_pose(theta, x, y, 'pick')).reshape(3,3,4)
            gp_rotated_middle = gp_rotated_global[:,1,:]

            # Choose the nearest point from the end effector
            a = 1.5  # slope of the line
            if which_side=='right':
                k = gp_rotated_middle[:,3] - a * gp_rotated_middle[:,2]  # y = a*x + k
                arg = np.argmin(k)
            elif which_side=='left':
                k = gp_rotated_middle[:, 3] + a * gp_rotated_middle[:, 2]  # y = -a*x + k
                arg = np.argmin(k)

            # Choose the farthest point from the peg among the three points on the same side
            argmax = np.argmax(np.linalg.norm(gp_rotated_global[arg][:,2:] - peg_points[n-1], axis=1))
            final_grasping_pose.append([n, int(gp_rotated_global[arg][argmax][0]), gp_rotated_global[arg][argmax][1],
                                  gp_rotated_global[arg][argmax][2], gp_rotated_global[arg][argmax][3]])
        return final_grasping_pose  # [peg number, gp number, theta, x, y]

    def find_placing_pose(self, final_grasping_pose, result_place_block, peg_points):
        final_placing_pose = []
        for res_gp, res in zip(final_grasping_pose, result_place_block):
            n, theta, x, y = res
            pp_rotated_global = np.array(self.get_sample_grasping_pose(theta, x, y, 'place'))
            gp_number = res_gp[1]
            final_placing_pose.append([n, gp_number, pp_rotated_global[gp_number][1], pp_rotated_global[gp_number][2], pp_rotated_global[gp_number][3]])
        return final_placing_pose # [peg number, gp number, theta, x, y]

    def overlay_blocks_and_pegs(self, img, peg_points):
        try:
            # Coloring yellow on blocks and pegs
            colored = self.change_color(img, (0, 255, 255))

            # Coloring pegs & put white letters numbering pegs
            count = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = [[-40,-20],[-40,-20],[-40,-20],[-40,-20],[-40,-20],[-40,-20],
                        [-20,-30],[-20,-30],[-20,-30],[-20,40],[-20,40],[-20,40]]
            for p,tp in zip(peg_points,text_pos):
                cv2.circle(colored, (p[0], p[1]), 5, (255, 0, 0), 2, -1)  # blue color overlayed on pegs
                # text = "%d" % (count);
                # cv2.putText(colored, text, (p[0]+tp[0], p[1]+tp[1]), font, 0.5, (255, 255, 255), 1)
                # count += 1
        except:
            pass
        return colored

    def overlay_block_contour(self, img, result_blocks):
        overlayed = np.copy(img)
        # Overlay contour
        for res in result_blocks:
            n, theta, x, y = res
            dx = self.__contour.shape[1]
            dy = self.__contour.shape[0]
            roi = overlayed[y:y + dy, x:x + dx]
            transformed = self.transform_img(self.__contour, theta, 0, 0)
            transformed_inv = cv2.bitwise_not(transformed)
            bg = cv2.bitwise_and(roi, roi, mask=transformed_inv)
            transformed_colored = self.change_color(transformed, (0, 255, 0))  # green color overlayed
            dst = cv2.add(bg, transformed_colored)
            overlayed[y:y + dy, x:x + dx] = dst
        return overlayed

    def overlay_pegs(self, img, peg_points, put_text=False):
        # Coloring white on blocks
        pegs_colored = self.change_color(img, (255, 255, 255))
        count = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        for p in peg_points:
            cv2.circle(pegs_colored, (p[0], p[1]), 5, (0, 255, 0), 2, -1)   # green color overlayed
            if put_text:
                text = "%d" % (count);
                cv2.putText(pegs_colored, text, (p[0]-10, p[1]-10), font, 0.5, (255, 255, 255), 1)
                count += 1
        return pegs_colored

    def overlay_grasping_pose(self, img, selected_grasping_pose_left, selected_grasping_pose_right):
        overlayed = np.copy(img)
        for gp in selected_grasping_pose_left:
            gp = list(map(int, gp))
            cv2.circle(overlayed, (gp[3], gp[4]), 3, (0, 0, 255), 2, -1)  # red color overlayed
        for gp in selected_grasping_pose_right:
            gp = list(map(int, gp))
            cv2.circle(overlayed, (gp[3], gp[4]), 3, (255, 255, 255), 2, -1)  # white color overlayed
        return overlayed

    def overlay_numbering(self, img, grasping_pose):
        overlayed = np.copy(img)
        count = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        for gp in grasping_pose:
            gp = gp.astype(int)
            text = "%d" % (count); cv2.putText(overlayed, text, (gp[0][1] - 20, gp[0][2]), font, 0.3, (255, 255, 255), 1)
            count += 1
            text = "%d" % (count); cv2.putText(overlayed, text, (gp[1][1] + 10, gp[1][2]), font, 0.3, (255, 255, 255), 1)
            count += 1
            text = "%d" % (count); cv2.putText(overlayed, text, (gp[2][1], gp[2][2] + 15), font, 0.3, (255, 255, 255), 1)
            count += 1
        return overlayed

    def change_color(self, img, color):
        colored = np.copy(img)
        colored = cv2.cvtColor(colored, cv2.COLOR_GRAY2BGR)
        args = np.argwhere(colored)
        for n in args:
            colored[n[0]][n[1]] = list(color)
        return colored

    def FLSPerception(self, img_depth):
        # Depth masking: thresholding by depth to find blocks & pegs
        blocks_masked = cv2.inRange(img_depth, self.__masking_depth_block[0], self.__masking_depth_block[1])
        pegs_masked = cv2.inRange(img_depth, self.__masking_depth_peg[0], self.__masking_depth_peg[1])

        cv2.imshow("", blocks_masked)
        cv2.imwrite('blocks_masked.png', blocks_masked)
        cv2.waitKey(0)

        # Pegs detection & overlay
        peg_points = self.pegs_detection(pegs_masked, self.__number_pegs)
        pegs_overlayed = self.overlay_pegs(pegs_masked, peg_points, True)

        # Add circled pegs on image of blocks
        blocks_added_pegs = cv2.add(blocks_masked, pegs_masked)

        # Find blocks from the segment images around peg_points
        result_global_blocks, result_global_wo_blocks = self.find_blocks(blocks_masked, peg_points)

        # Classify result of the blocks
        pickup_blocks_rarm, place_blocks_rarm = self.classify_blocks(result_global_blocks, result_global_wo_blocks)

        # Find grasping pose
        all_grasping_pose = self.find_all_grasping_pose(result_global_blocks)
        final_gp_rarm = self.find_grasping_pose(pickup_blocks_rarm, peg_points, 'right')
        final_pp_rarm = self.find_placing_pose(final_gp_rarm, place_blocks_rarm, peg_points)

        # Coloring & Overlay
        blocks_overlayed = self.overlay_blocks_and_pegs(blocks_added_pegs, peg_points)
        blocks_overlayed = self.overlay_block_contour(blocks_overlayed, pickup_blocks_rarm)
        blocks_overlayed = self.overlay_block_contour(blocks_overlayed, place_blocks_rarm)
        blocks_overlayed = self.overlay_grasping_pose(blocks_overlayed, final_gp_rarm, final_gp_rarm)
        blocks_overlayed = self.overlay_grasping_pose(blocks_overlayed, final_pp_rarm, final_pp_rarm)
        # blocks_overlayed = self.overlay_numbering(blocks_overlayed, all_grasping_pose)
        return final_gp_rarm, final_pp_rarm, peg_points, pegs_overlayed, blocks_overlayed

if __name__ == '__main__':
    # filename = "../img/img_depth"
    # img_depth = cv2.imread(filename, cv2.IMREAD)
    # img_depth = np.zeros((400,300), np.uint8)
    # bd = BlockDetection()
    # bd.FLSPerception(img_depth)

    from FLSpegtransfer.vision.ZividCapture import ZividCapture
    BD = BlockDetection()
    zivid = ZividCapture()
    while True:
        try:
            zivid.capture_3Dimage()
            img_color, img_depth, img_point = BD.img_crop(zivid.image, zivid.depth, zivid.point)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
            img_color_org = np.copy(img_color)

            # Find balls and overlay
            final_gp_rarm, final_pp_rarm, peg_points, pegs_overlayed, blocks_overlayed = BD.FLSPerception(img_depth)

            cv2.imshow("color", img_color)
            cv2.imshow("2", pegs_overlayed)
            cv2.imshow("3", blocks_overlayed)
            key = cv2.waitKey(1)
        finally:
            pass