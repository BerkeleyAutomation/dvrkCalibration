import numpy as np
from FLSpegtransfer.motion.dvrkMotionBridgeP import dvrkMotionBridgeP
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics

class dvrkCalibratedMotion():
    def __init__(self, model='None'):

        # load model
        self.model = model
        if self.model=='neural net':
            from FLSpegtransfer.motion.NNModel import NNModel
            self.NN = NNModel(ensemble=3, horizon=4, iter=3, alpha=0.5, model='forward')
        elif self.model=='linear':
            from FLSpegtransfer.motion.LinearModel import LinearModel
            self.LM = LinearModel(4)

        # define instances
        self.dvrk = dvrkMotionBridgeP()
        self.dvrk_model = dvrkKinematics()

        # limit joints
        self.joint_limits_min = np.deg2rad([-80, -60, 0.08, -100, -90, -100])
        self.joint_limits_min[2] = 0.08
        self.joint_limits_max = np.deg2rad([80, 60, 0.245, 100, 90, 100])
        self.joint_limits_max[2] = 0.245

    def set_pose(self, pos, rot, jaw):
        joint = self.dvrk_model.pose_to_joint(pos, rot)
        self.set_joint(joint, jaw)

    def set_joint(self, q_target, jaw=[]):
        q_cmd = self.calibrate(q_target)
        self.dvrk.set_joint(joint1=q_cmd, jaw1=jaw)

    def calibrate(self, q_target):
        if self.model == 'neural net':
            q_cmd = self.NN.step(q_target)
        elif self.model == 'linear':
            q_cmd = self.LM.step(q_target)
        else:
            q_cmd = q_target
        self.verify_joints(q_cmd)
        return q_cmd

    def verify_joints(self, q_cmd):
        for i,q in enumerate(q_cmd):
            if q > self.joint_limits_max[i]:
                q_cmd[i] = self.joint_limits_max[i]
                print ("q",i+1,"=", q, " is limited by max, new value=", q_cmd[i])
            elif q < self.joint_limits_min[i]:
                q_cmd[i] = self.joint_limits_min[i]
                print("q", i+1,"=", q, " is limited by min, new value=", q_cmd[i])


if __name__ == "__main__":
    CM = dvrkCalibratedMotion(model='neural net')
