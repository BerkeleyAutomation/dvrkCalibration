import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import time

from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection import BlockDetection
from FLSpegtransfer.motion.dvrkPegTransferMotion import dvrkPegTransferMotion

class FLSPegTransfer():
    def __init__(self):
        # import modules
        self.BD = BlockDetection()
        self.dvrk_motion = dvrkPegTransferMotion()
        self.zivid = ZividCapture()

        # load transform
        self.Trc = np.load('calibration_files/Trc.npy')

        # data members
        self.img_color = []
        self.img_depth = []
        self.img_point = []
        self.moving_dir = 'l2r'
        self.moving_order = 0
        self.auto_flag = True
        self.main()

    def move_blocks(self, grasping_pose):
        if not grasping_pose == [[]]*2:
            Rrc = self.Trc[:3, :3]  # transform
            trc = self.Trc[:3,-1]
            for gp in grasping_pose:
                n, ang, Xc, Yc, Zc, seen = gp
                x,y,z = Rrc.dot([Xc,Yc,Zc]) + trc  # position in terms of the robot's coordinate
                gp1 = [x,y,z, ang]
                # gp2 =
                if seen == True:
                    self.dvrk_motion.pickup_block(gp1, [])
                else:
                    self.dvrk_motion.place_block(gp1, [])

    def decide_action(self, gp):
        while True:
            if self.moving_dir == 'l2r' and self.moving_order == 6:  # switching moving direction
                self.moving_dir = 'r2l'
                self.moving_order = 0
                gp_selected = []
                cmd = 'continue'
                break
            elif self.moving_dir == 'r2l' and self.moving_order == 6:  # terminate
                exit()
            else:
                if gp[self.moving_order * 2] == []:
                    self.moving_order += 1
                else:
                    gp_selected = [gp[self.moving_order * 2], gp[self.moving_order * 2 + 1]]
                    cmd = 'move'
                    self.moving_order += 1
                    break
        return cmd, gp_selected

    def main(self):
        try:
            # random motion
            self.dvrk_motion.move_random()

            # repeat until peg transfer is finished
            while True:
                # move to origin
                self.dvrk_motion.move_origin()
                time.sleep(0.3)

                # capture
                # image = np.load('record/image.npy')
                # depth = np.load('record/depth.npy')
                # point = np.load('record/point.npy')
                # img_color, self.img_depth, self.img_point = self.BD.img_crop(image, depth, point)
                self.zivid.capture_3Dimage()
                img_color, self.img_depth, self.img_point = self.BD.img_crop(self.zivid.image, self.zivid.depth, self.zivid.point)
                self.img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

                # scanning
                # [grasping pose, block poses, image pegs, image blocks]
                gp, img_pegs_ovl, img_blks_ovl = self.BD.FLSPerception(self.img_depth, self.img_point, self.moving_dir)

                # image display
                cv2.imshow("img_original", self.img_color)
                cv2.imshow("img_pegs", img_pegs_ovl)
                cv2.imshow("img_blocks", img_blks_ovl)
                cv2.waitKey(100)

                # decide action
                cmd, gp = self.decide_action(gp)

                # take action
                print(self.moving_dir, self.moving_order, gp)
                if cmd == 'continue':
                    continue
                elif cmd == 'move':
                    self.move_blocks(gp)

        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    FLSPegTransfer()