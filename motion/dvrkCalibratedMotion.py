# from google.colab import drive
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
import copy

import sys
for p in sys.path:
  if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BallDetection import BallDetection

import FLSpegtransfer.utils.CmnUtil as U
root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'


class CalibratedMotion():
    def __init__(self, model_dir, motion_bridge=None):
      if not motion_bridge:
          from FLSpegtransfer.motion.dvrkMotionBridgeP import dvrkMotionBridgeP
          self.dvrk = dvrkMotionBridgeP()
      else:
          self.dvrk = motion_bridge

      self.zivid = ZividCapture()
      self.BD = BallDetection()

      # self.num_ensemble_models = 0
      # self.num_ensemble_models = 10
      self.num_ensemble_models = 3

      self.use_inverse_model = True
      self.use_delta = False
      self.use_sampling = False
      # self.use_model_output = False
      self.use_linear = False

      if self.use_linear:
          from FLSpegtransfer.motion.LinearModel import LinearModel
          self.linear_model = LinearModel(4)

      # self.horizon = 4
      # self.horizon = 6
      self.horizon = 5
      self.models = self.load_models(model_dir)
      self.movement_cache = []
      self.num_to_sample = 8
      self.num_delta_iters = 3
      self.alpha = 0.5
      # self.alpha = 1

      # 1.
        # create new traj w
        # num_delta_iters: 1, 3, 10
        # alpha: .5, 1
      self.sampling_level = 1

      self.joint_limits = {4:80, 5:60, 6:60}

    def move(self, desired):
        self.add_to_cache(desired)
        q_cmd = self.get_next_cmd()
        # if self.use_model_output and np.all(q_cmd != desired):
        #     self.movement_cache[-1] = self.invoke_model(q_cmd)
        #     print(self.movement_cache[-1])
        # else:
        self.movement_cache[-1] = q_cmd
        self.move_arm(q_cmd)
        return q_cmd

    def get_next_cmd(self):
        if len(self.movement_cache) < self.horizon and not self.use_linear:
            print('not using history')
            return self.get_desired_pos()
        elif self.use_sampling:
            return self.get_joints_from_sampling(self.sampling_level)
        elif self.use_delta:
            print('using history')
            return self.get_delta_cmd()
        elif self.use_inverse_model:
            q123 = self.get_desired_pos()[:3]
            q456 = self.get_inverse_cmd()
            joints = np.concatenate((q123, q456))
            return joints
        elif self.use_linear:
            return self.linear_model.step(self.get_desired_pos())
        else:
            return self.get_desired_pos()


    def add_to_cache(self, desired):
      if len(self.movement_cache) < self.horizon:
        self.movement_cache.append(desired)
      else:
        self.movement_cache.pop(0)
        self.movement_cache.append(desired)

    def invoke_model(self, q_cmd):
      inp = self.get_movement_cache_copy()
      inp[-1] = q_cmd
      inp = [d[3:] for d in inp]
      inp = np.expand_dims(inp, axis=0)
      outputs = []
      for model in self.models:
          outputs.append(model.predict(inp)[0])
      print("STD:", np.std(outputs))
      # print("MED:", np.median(outputs, axis=0))
      # print("AVG:", np.mean(outputs, axis=0))

      output = np.median(outputs, axis=0)
      return output

    def load_models(self, model_dir):
        models = []
        for i in range(1, self.num_ensemble_models + 1):
            if self.use_sampling and i >= 4:
                continue
            json_file = open(model_dir + str(i) + '/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_dir + str(i) + "/model.h5")
            models.append(loaded_model)

        print("Loaded " + str(len(models)) + " models from disk.")
        return models

    def get_inverse_cmd(self):
        assert not self.use_delta
        q_cmd = self.get_desired_pos()
        return self.invoke_model(q_cmd)

    def get_delta_cmd(self):
        assert not self.use_inverse_model
        q_target = self.get_desired_pos()
        q_cmd = self.get_desired_pos()
        for i in range(self.num_delta_iters):
            # self.alpha = (float(self.num_delta_iters) - float(i)) / float(self.num_delta_iters)
            q_cmd = self.get_corrected_cmd(q_cmd, q_target)
        # print("orig cmd:", np.round(q_target, 8))
        # print("new cmd :", np.round(q_cmd, 8))
        # print()
        return q_cmd

        # q_des + e_h = q_act
        # (q_des - e_h) + e_h = q_act
        # q_des = q_act (!!)

        # q_des = q_cmd0
        # q_cmd0 + e_h = q_act0

        # (q_cmd0 - e_h) + e_h' = q_act1
        # (q_cmd1) + e_h'= q_act1

        # (q_cmd1 - e_h') + e_h''= q_act2

        # q_cmd0 - e_h - e_h'

        # q_des = q_act (!!)

    def get_corrected_cmd(self, q_cmd, q_target):
        e_h = np.concatenate(([0., 0., 0.], self.invoke_model(q_cmd))) + q_cmd - q_target
        q_cmd = q_cmd - self.alpha * e_h
        return q_cmd

    def get_joints_from_sampling(self, level):
      # Use random sampling in 3 mm by 3 mm box
      q_des = np.array(self.get_desired_pos())

      min_loss = float('inf')
      best_cmd = None

      angle_one = 24

      # Search One
      q4_angles = self.get_sampled_angles(angle_one, self.num_to_sample)
      q5_angles = self.get_sampled_angles(angle_one, self.num_to_sample)
      q6_angles = self.get_sampled_angles(angle_one, self.num_to_sample)

      # print(q4_angles)
      # print(q5_angles)
      # print(q6_angles)

      zeros = np.zeros(3)
      for x, q4 in enumerate(q4_angles):
        for y, q5 in enumerate(q5_angles):
          for z, q6 in enumerate(q6_angles):
            q_cmd = q_des + np.concatenate((zeros, [q4, q5, q6]))
            if not self.verify_joints(q_cmd):
                continue

            e_h = np.array(self.invoke_model(q_cmd))
            to_add = np.concatenate((zeros, e_h))
            q_est = q_cmd + to_add

            mse_dist = mean_squared_error(q_des, q_est)
            if mse_dist < min_loss:
              min_loss = mse_dist
              best_cmd = np.copy(q_cmd)
              # print("new best MSE:", min_loss)
              # print(x, y, z)

      if level == 1:
          return best_cmd

      angle_two = 2 * angle_one / float(self.num_to_sample)
      # Search Two
      q4_angles = self.get_sampled_angles(angle_two, self.num_to_sample)
      q5_angles = self.get_sampled_angles(angle_two, self.num_to_sample)
      q6_angles = self.get_sampled_angles(angle_two, self.num_to_sample)

      # print(q4_angles)
      # print(q5_angles)
      # print(q6_angles)

      zeros = np.zeros(3)
      for q4 in q4_angles:
        for q5 in q5_angles:
          for q6 in q6_angles:
            q_cmd = best_cmd + np.concatenate((zeros, [q4, q5, q6]))
            if not self.verify_joints(q_cmd):
                continue

            e_h = np.array(self.invoke_model(q_cmd))
            to_add = np.concatenate((zeros, e_h))
            q_est = q_cmd + to_add

            mse_dist = mean_squared_error(q_des, q_est)
            if mse_dist < min_loss:
              min_loss = mse_dist
              best_cmd_2 = np.copy(q_cmd)
              # print(q_cmd[3:])
              # print("new best MSE:", min_loss)

      angle_three = 2 * angle_two / float(self.num_to_sample)
      # Search Three
      q4_angles = self.get_sampled_angles(angle_three, self.num_to_sample)
      q5_angles = self.get_sampled_angles(angle_three, self.num_to_sample)
      q6_angles = self.get_sampled_angles(angle_three, self.num_to_sample)

      # print(q4_angles)
      # print(q5_angles)
      # print(q6_angles)

      if level == 2:
          return best_cmd_2

      zeros = np.zeros(3)
      for q4 in q4_angles:
        for q5 in q5_angles:
          for q6 in q6_angles:
            q_cmd = best_cmd_2 + np.concatenate((zeros, [q4, q5, q6]))
            if not self.verify_joints(q_cmd):
                continue

            e_h = np.array(self.invoke_model(q_cmd))
            to_add = np.concatenate((zeros, e_h))
            q_est = q_cmd + to_add

            mse_dist = mean_squared_error(q_des, q_est)
            if mse_dist < min_loss:
              min_loss = mse_dist
              best_cmd_3 = np.copy(q_cmd)
              # print("new best MSE:", min_loss)

      # print("q_des: ", q_des)
      e_h = np.array(self.invoke_model(q_des))
      to_add = np.concatenate((zeros, e_h))
      q_est = q_des + to_add
      # print("f(q_des):  ", q_est)
      # print("MSE:", mean_squared_error(q_des, q_est))
      # print()

      if best_cmd_3 is not None:
        q_cmd = best_cmd_3
      elif best_cmd_2 is not None:
        q_cmd = best_cmd_2
      else:
        q_cmd = best_cmd

      # print("Q_CMD:", q_cmd)

      # print("q_cmd:     ", best_cmd_3)
      e_h = np.array(self.invoke_model(q_cmd))
      to_add = np.concatenate((zeros, e_h))
      q_est = q_cmd + to_add
      #print("f(q_cmd):  ", q_est)
      #print("MSE:", mean_squared_error(q_des, q_est))
      # print()
      # print()
      return q_cmd

    def move_arm(self, joints):
      self.dvrk.set_joint(joint1=joints)

    def get_desired_pos(self):
      return copy.deepcopy(self.movement_cache[-1])

    def get_movement_cache_copy(self):
      return copy.deepcopy(self.movement_cache)

    def verify_joints(self, q_cmd):
        _, _, _, q4, q5, q6 = q_cmd
        if q4 <= -np.deg2rad(self.joint_limits[4]) or q4 >= np.deg2rad(self.joint_limits[4]):
            return False
        if q5 <= -np.deg2rad(self.joint_limits[5]) or q5 >= np.deg2rad(self.joint_limits[5]):
            return False
        if q6 <= -np.deg2rad(self.joint_limits[6]) or q6 >= np.deg2rad(self.joint_limits[6]):
            return False
        return True

    def get_sampled_angles(self, deg, steps):
      rads = np.deg2rad(deg)
      upper = rads * 2
      step = upper / float(steps)
      arr = [i * step for i in range(steps + 1)]
      rads_to_sample = np.array(arr) - rads
      return rads_to_sample


# Delta
    # iter 1, 2, 3
# use sampling
    # level 1, 2, 3

    def verification(self, rand=False, peg=False):
        try:
            root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
            file_path = root + 'experiment/0_trajectory_extraction/'
            if rand:
                traj = np.load(file_path + 'verification_traj_random_sampling_100.npy')
            elif peg:
                traj = np.load(file_path + 'verification_traj_insertion_50.npy')
                print(np.shape(traj))
            else:
                traj = np.load(file_path + 'verification_traj_random.npy')

            jaw1 = [5. * np.pi / 180.]
            self.dvrk.set_pose(jaw1=jaw1)

            time_st = time.time()   # (sec)
            time_stamp = []
            new_q_act = []
            new_q_des = []
            for qd1,qd2,qd3,qd4,qd5,qd6 in traj:
                # qd1 = self.dvrk.act_joint1[0]
                # qd2 = self.dvrk.act_joint1[1]
                # qd3 = self.dvrk.act_joint1[2]
                joint_new = CB.move([qd1, qd2, qd3, qd4, qd5, qd6])
                self.dvrk.set_joint(joint1=joint_new)
                time.sleep(0.3)

                # Capture image from Zivid
                self.zivid.capture_3Dimage()
                img_color, img_depth, img_point = self.BD.img_crop(self.zivid.image, self.zivid.depth, self.zivid.point)
                img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
                img_color_org = np.copy(img_color)

                # Find balls
                pbs = self.BD.find_balls(img_color_org, img_depth, img_point)
                img_color = self.BD.overlay_balls(img_color, pbs)

                # Find tool position, joint angles, and overlay
                if pbs[0] == [] or pbs[1] == []:
                    qa1=0.0; qa2=0.0; qa3=0.0; qa4=0.0; qa5=0.0; qa6=0.0
                else:
                    # Find tool position, joint angles, and overlay
                    pt = self.BD.find_tool_position(pbs[0], pbs[1])  # tool position of pitch axis
                    pt = np.array(pt) * 0.001  # (m)

                    pt = self.BD.Rrc.dot(pt) + self.BD.trc
                    qa1, qa2, qa3 = self.BD.ik_position(pt)

                    # Find tool orientation, joint angles, and overlay
                    count_pbs = [pbs[2], pbs[3], pbs[4], pbs[5]]
                    if count_pbs.count([]) >= 2:
                        qa4=0.0; qa5=0.0; qa6=0.0
                    else:
                        Rm = self.BD.find_tool_orientation(pbs[2], pbs[3], pbs[4], pbs[5])  # orientation of the marker
                        qa4, qa5, qa6 = self.BD.ik_orientation(qa1, qa2, Rm)
                        img_color = self.BD.overlay_tool(img_color, [qa1, qa2, qa3, qa4, qa5, qa6], (0, 255, 0))

                # Append data pairs
                # joint angles
                new_q_act.append([qa1, qa2, qa3, qa4, qa5, qa6])
                new_q_des.append(joint_new)
                time_stamp.append(time.time() - time_st)
                print('index: ', len(new_q_des),'/',len(traj))
                print('t_stamp: ', time.time() - time_st)
                print ('q_des: ', [qd1,qd2,qd3,qd4,qd5,qd6])
                print('new_q_des: ', joint_new)
                print('new_q_act: ', [qa1, qa2, qa3, qa4, qa5, qa6])
                print(' ')

                # Visualize
                cv2.imshow("images", img_color)
                cv2.waitKey(1) & 0xFF
                # cv2.waitKey(0)

        finally:
            np.save('new_q_des', new_q_des)
            np.save('new_q_act', new_q_act)
            np.save('t_stamp', time_stamp)
            print("Data is successfully saved")

if __name__ == "__main__":
    # model_dir = "../experiment/4_verification/model/RNN_forward_peg_traj/"
    model_dir = "../experiment/4_verification/model/RNN_inverse_peg_traj/"
    # model_dir = "../experiment/4_verification/model/RNN_forward_random_traj/"
    # model_dir = "../experiment/4_verification/model/RNN_inverse_random_sampled_traj_short/"
    CM = CalibratedMotion(model_dir)
    CM.verification(rand=False, peg=True)