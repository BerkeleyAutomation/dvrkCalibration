from FLSpegtransfer.motion.dvrkArm import dvrkArm
import time
import numpy as np

def record_trajectory():
    try:
        time_curr = 0.0
        time_st = time.time()
        while time_curr < 1190:
            time_curr = time.time() - time_st
            joint1 = p1.get_current_joint(wait_callback=True)
            time.sleep(0.1)
            joint1_record.append(joint1)
            print(joint1, time_curr)
    finally:
        np.save("training_traj_peg_transfer", joint1_record)
        print("Data is successfully saved")

if __name__ == "__main__":
    p1 = dvrkArm('/PSM1')
    joint1_record = []
    record_trajectory()