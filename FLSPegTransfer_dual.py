import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import time

from FLSpegtransfer.vision.BlockDetection_dual import BlockDetection
from FLSpegtransfer.vision.MappingC2R import MappingC2R
from FLSpegtransfer.vision.ZividCapture import ZividCapture
# from FLSpegtransfer.motion.dvrkBlockTransfer import dvrkBlockTransfer
import FLSpegtransfer.utils.CmnUtil as U

class FLSPegTransfer():
    def __init__(self):
        # import other modules
        row_board = 6
        col_board = 8
        filename = 'calibration_files/mapping_table_PSM1'
        self.__mapping1 = MappingC2R(filename, row_board, col_board)
        filename = 'calibration_files/mapping_table_PSM2'
        self.__mapping2 = MappingC2R(filename, row_board, col_board)
        self.__block_detection = BlockDetection()
        # self.__dvrk = dvrkBlockTransfer()
        self.__zivid = ZividCapture()

        # data members
        self.__img_color = []
        self.__img_depth = []
        self.__points_list = []

        self.__moving_l2r_flag = True
        self.__moving_r2l_flag = False

        self.__pos_offset1 = [0.0, 0.0, 0.0]    # offset in (m)
        self.__pos_offset2 = [0.0, 0.0, 0.0]

        # self.interval_ms = 300
        self.main()

    def __img_crop(self, img):
        # Image cropping
        x = 710; w = 520
        y = 450; h = 400
        cropped = img[y:y + h, x:x + w]
        return cropped

    def select_ordering(self, final_gp_larm, final_gp_rarm, direction):
        if direction == 'l2r':
            n_larm = np.array(final_gp_larm)[:,0]
            n_rarm = np.array(final_gp_rarm)[:,0]
            n_larm = map(int, n_larm[n_larm<=6])
            n_rarm = map(int, n_rarm[n_rarm<=6])
        elif direction == 'r2l':
            n_larm = np.array(final_gp_larm)[:,0]
            n_rarm = np.array(final_gp_rarm)[:,0]
            n_larm = map(int, n_larm[n_larm>6])
            n_rarm = map(int, n_rarm[n_rarm>6])

        n_larm = np.pad(n_larm, pad_width=(0, max(0, len(n_rarm)-len(n_larm))), mode='constant', constant_values=(0,0))
        n_rarm = np.pad(n_rarm, pad_width=(0, max(0, len(n_larm)-len(n_rarm))), mode='constant', constant_values=(0,0))
        return n_larm, n_rarm

    def move_blocks(self, pick_number_larm, pick_number_rarm, final_gp_larm, final_pp_larm, final_gp_rarm, final_pp_rarm):
        arg_pick = np.argwhere(np.array(final_gp_rarm)[:, 0] == pick_number_rarm)
        arg_place = arg_pick
        if len(arg_pick) == 0 or len(arg_place) == 0:
            pos_pick1 = []
            rot_pick1 = []
            pos_place1 = []
            rot_place1 = []
        else:
            arg_pick = arg_pick[0][0]
            arg_place = arg_place[0][0]
            pos_pick1 = self.__mapping1.transform_pixel2robot(final_gp_rarm[arg_pick][3:], final_gp_rarm[arg_pick][2])
            rot_pick1 = [final_gp_rarm[arg_pick][2], 0, 0]
            pos_place1 = self.__mapping1.transform_pixel2robot(final_pp_rarm[arg_place][3:], final_pp_rarm[arg_place][2])
            rot_place1 = [final_pp_rarm[arg_place][2], 0, 0]

        arg_pick = np.argwhere(np.array(final_gp_larm)[:, 0] == pick_number_larm)
        arg_place = arg_pick
        if len(arg_pick) == 0 or len(arg_place) == 0:
            pos_pick2 = []
            rot_pick2 = []
            pos_place2 = []
            rot_place2 = []
        else:
            arg_pick = arg_pick[0][0]
            arg_place = arg_place[0][0]
            pos_pick2 = self.__mapping2.transform_pixel2robot(final_gp_larm[arg_pick][3:], final_gp_larm[arg_pick][2])
            rot_pick2 = [final_gp_larm[arg_pick][2], 0, 0]
            pos_place2 = self.__mapping2.transform_pixel2robot(final_pp_larm[arg_place][3:], final_pp_larm[arg_place][2])
            rot_place2 = [final_pp_larm[arg_place][2], 0, 0]

        which_arm = 'Both'
        pos_pick1 = [pos_pick1[0] + self.__pos_offset1[0], pos_pick1[1] + self.__pos_offset1[1],
                     pos_pick1[2] + self.__pos_offset1[2]]
        pos_pick2 = [pos_pick2[0] + self.__pos_offset2[0], pos_pick2[1] + self.__pos_offset2[1],
                     pos_pick2[2] + self.__pos_offset2[2]]
        # self.__dvrk.pickup(pos_pick1=pos_pick1, rot_pick1=rot_pick1, pos_pick2=pos_pick2, rot_pick2=rot_pick2, which_arm=which_arm)
        # self.__dvrk.place(pos_place1=pos_place1, rot_place1=rot_place1, pos_place2=pos_place2, rot_place2=rot_place2, which_arm=which_arm)

    def main(self):
        try:
            user_input = input("Are you going to proceed automatically? (y or n)")
            if user_input == "y":   auto_flag = True
            elif user_input == "n": auto_flag = False
            else:   return
            while True:
                # Capture image from zivid
                self.__zivid.capture_3Dimage()
                self.__img_color = self.__img_crop(self.__zivid.image)
                self.__img_depth = self.__img_crop(self.__zivid.depth)
                if self.__img_color == [] or self.__img_depth == []:
                    pass
                else:
                    # Scanning
                    # self.__dvrk.move_origin()
                    time.sleep(0.3)

                    # Perception output
                    final_gp_larm, final_pp_larm, final_gp_rarm, final_pp_rarm, peg_points, pegs_overlayed, blocks_overlayed\
                        = self.__block_detection.FLSPerception(self.__img_depth)

                    cv2.imshow("img_color", self.__img_color)
                    cv2.imshow("masked_pegs", pegs_overlayed)
                    cv2.imshow("masked_blocks", blocks_overlayed)
                    cv2.waitKey(1000)
                    if not auto_flag:
                        user_input = input("1: Left to right,  2: Right to left")
                        if user_input == "1":
                            self.__moving_l2r_flag = True
                            self.__moving_r2l_flag = False
                        elif user_input == "2":
                            self.__moving_l2r_flag = False
                            self.__moving_r2l_flag = True
                        else:
                            self.__moving_l2r_flag = False
                            self.__moving_r2l_flag = False

                    # Move blocks from left to right
                    if self.__moving_l2r_flag:
                        n_larm, n_rarm = self.select_ordering(final_gp_larm, final_gp_rarm, direction='l2r')
                        if auto_flag:   # check if completed
                            if len(n_larm)==0 and len(n_rarm)==0:
                                self.__moving_l2r_flag = False
                                self.__moving_r2l_flag = True
                        for nl, nr in zip(n_larm, n_rarm):
                            self.move_blocks(nl, nr, final_gp_larm, final_pp_larm, final_gp_rarm, final_pp_rarm)
                    # Move blocks from right to left
                    elif self.__moving_r2l_flag:
                        n_larm, n_rarm = self.select_ordering(final_gp_larm, final_gp_rarm, direction='r2l')
                        if auto_flag:
                            if len(n_larm)==0 and len(n_rarm)==0:
                                self.__moving_l2r_flag = False
                                self.__moving_r2l_flag = False
                        for nl, nr in zip(n_larm, n_rarm):
                            self.move_blocks(nl, nr, final_gp_larm, final_pp_larm, final_gp_rarm, final_pp_rarm)
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    FLSPegTransfer()