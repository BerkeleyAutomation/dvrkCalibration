import numpy as np
from FLSpegtransfer.vision.MappingC2R import MappingC2R
from FLSpegtransfer.motion.dvrkDualArm import dvrkArm
import FLSpegtransfer.utils.CmnUtil as U
import time

def move_to_corners(arm_name, roll_angle=0):
    row_board = 6
    col_board = 8

    pos_org = [[0.055, 0.0, -0.1], [-0.055, 0.0, -0.1]]
    rot_org1 = np.array([roll_angle, 0, 0]) * np.pi / 180.
    rot_org2 = np.array([roll_angle, 0, 0]) * np.pi / 180.
    rot_org = [U.euler_to_quaternion(rot_org1), U.euler_to_quaternion(rot_org2)]
    jaw_org = [[0 * np.pi / 180.], [0 * np.pi / 180.]]
    ready_height = 0.03

    arm = dvrkArm(arm_name)
    if arm_name=='/PSM1':
        filename = '../calibration_files/mapping_table_PSM1'
        index = 0
    elif arm_name=='/PSM2':
        filename = '../calibration_files/mapping_table_PSM2'
        index = 1

    mapping = MappingC2R(filename, row_board, col_board)

    # Move PSM1 to go through all points.
    arm.set_pose(pos_org[index], rot_org[index])
    arm.set_jaw(jaw_org[index])
    for i in range(4,row_board):
        for j in range(4,col_board):
            if (index==0 and j==0) or (index==0 and j==1):    # PSM1
                pass
            elif (index==1 and j==6) or (index==1 and j==7):   # PSM2
                pass
            else:
                pos_0 = mapping.mapping_table_0[i][j]
                pos_90 = mapping.mapping_table_90[i][j]
                pos = interpolate(pos_0, pos_90, roll_angle)
                print pos

                # just checking if the ROS input is fine
                # user_input = raw_input("Are you sure the values to input to the robot arm?(y or n)")
                # if user_input == "y":
                arm.set_pose([pos[0], pos[1], pos[2]+ready_height], rot_org[index])
                arm.set_jaw(jaw_org[index])
                arm.set_pose(pos, rot_org[index])
                user_input = raw_input("Keep going?")
                if user_input == 'n':
                    arm.set_pose([pos[0], pos[1], pos[2] + ready_height], rot_org[index])
                    arm.set_pose(pos_org[index], rot_org[index])
                    exit(0)
                time.sleep(0.3)
                arm.set_pose([pos[0], pos[1], pos[2]+ready_height], rot_org[index])
                arm.set_jaw(jaw_org[index])
    arm.set_pose(pos_org[index], rot_org[index])
    arm.set_jaw(jaw_org[index])

def interpolate(output_0, output_90, roll_angle):
    # interpolate the outputs according to the given roll angle
    roll_angle = abs(roll_angle)
    if roll_angle > 90:
        roll_angle -= 180
        roll_angle = abs(roll_angle)
    print roll_angle
    output_final = output_0 + (roll_angle - 0) * (output_90 - output_0) / (90 - 0)
    return output_final

if __name__ == "__main__":
    # move_to_corners(arm_name='/PSM2', roll_angle=0)
    # move_to_corners(arm_name='/PSM2', roll_angle=90)
    # move_to_corners(arm_name='/PSM1', roll_angle=0)
    move_to_corners(arm_name='/PSM1', roll_angle=90)