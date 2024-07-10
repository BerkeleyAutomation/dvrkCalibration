import numpy as np
import threading
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Float64
import dvrk.utils.CmnUtil as U
import dvrk.motion.dvrkVariables as dvrkVar
from dvrk.motion.dvrkKinematics import dvrkKinematics
import time
import rospy


class dvrkArm(object):
    """Simple arm API wrapping around ROS messages"""

    def __init__(self, arm_name, ros_namespace="/dvrk"):
        # continuous publish from dvrk_bridge
        # actual(current) values
        self.__act_joint = []
        self.__act_jaw = []
        self.__act_motor_current = [0.0] * 7

        # data members, event based
        self.__arm_name = arm_name
        self.__ros_namespace = ros_namespace
        self.__goal_reached_event = threading.Event()
        # self.__get_position_event = threading.Event()
        self.__get_joint_event = threading.Event()
        self.__get_jaw_event = threading.Event()
        self.__get_motor_current_event = threading.Event()

        self.__sub_list = []
        self.__pub_list = []

        # publisher
        self.__full_ros_namespace = self.__ros_namespace + self.__arm_name
        self.__set_position_joint_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_position_joint", JointState, latch=True, queue_size=1
        )
        self.__set_position_goal_joint_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_position_goal_joint", JointState, latch=True, queue_size=1
        )
        self.__set_position_cartesian_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_position_cartesian", Pose, latch=True, queue_size=1
        )
        self.__set_position_goal_cartesian_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_position_goal_cartesian", Pose, latch=True, queue_size=1
        )
        self.__set_position_jaw_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_position_jaw", JointState, latch=True, queue_size=1
        )
        self.__set_position_goal_jaw_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_position_goal_jaw", JointState, latch=True, queue_size=1
        )
        self.__set_velocity_ratio_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_joint_velocity_ratio", Float64, latch=True, queue_size=1
        )
        self.__set_acceleration_ratio_pub = rospy.Publisher(
            self.__full_ros_namespace + "/set_joint_acceleration_ratio", Float64, latch=True, queue_size=1
        )

        self.__pub_list = [
            self.__set_position_joint_pub,
            self.__set_position_goal_joint_pub,
            self.__set_position_cartesian_pub,
            self.__set_position_goal_cartesian_pub,
            self.__set_position_jaw_pub,
            self.__set_position_goal_jaw_pub,
            self.__set_velocity_ratio_pub,
            self.__set_acceleration_ratio_pub,
        ]

        self.__sub_list = [
            rospy.Subscriber(self.__full_ros_namespace + "/goal_reached", Bool, self.__goal_reached_cb),
            rospy.Subscriber(
                self.__full_ros_namespace + "/state_joint_current", JointState, self.__position_joint_current_cb
            ),
            rospy.Subscriber(
                self.__full_ros_namespace + "/state_jaw_current", JointState, self.__position_jaw_current_cb
            ),
            rospy.Subscriber(
                self.__full_ros_namespace + "/io/actuator_current_measured",
                JointState,
                self.__motor_current_measured_cb,
            ),
        ]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node("dvrkArm_node", anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

        print("dvrk", arm_name, "initialized")
        # wait until these are not empty
        # self.__act_joint = self.get_current_joint(wait_callback=True)
        # self.__act_jaw = self.get_current_jaw(wait_callback=True)

    def shutdown(self):
        rospy.signal_shutdown("Shutdown signal received.")

    """
    Callback function
    """

    def __goal_reached_cb(self, data):
        self.__goal_reached = data.data
        self.__goal_reached_event.set()

    def __position_joint_current_cb(self, data):
        self.__act_joint = list(data.position)
        self.__get_joint_event.set()

    def __position_jaw_current_cb(self, data):
        self.__act_jaw = list(data.position)
        self.__get_jaw_event.set()

    def __motor_current_measured_cb(self, data):
        self.__act_motor_current = list(data.position)
        self.__get_motor_current_event.set()

    """
    Get function
    """

    def get_current_pose(self, wait_callback=True):
        joint = self.get_current_joint(wait_callback=wait_callback)
        return dvrkKinematics.joint_to_pose(joint)

    def get_current_joint(self, wait_callback=True):
        if wait_callback:
            self.__get_joint_event.clear()
            if self.__get_joint_event.wait():  # 20 seconds at most
                return self.__act_joint
            else:
                return []
        else:
            return self.__act_joint

    def get_current_jaw(self, wait_callback=True):
        if wait_callback:
            self.__get_jaw_event.clear()
            if self.__get_jaw_event.wait(20):  # 20 seconds at most
                jaw = self.__act_jaw
                return jaw
            else:
                return []
        else:
            jaw = self.__act_jaw
            return jaw

    def get_motor_current(self, wait_callback=True):
        if wait_callback:
            self.__get_motor_current_event.clear()
            if self.__get_motor_current_event.wait(20):  # 20 seconds at most
                motor_current = self.__act_motor_current
                return motor_current
            else:
                return []
        else:
            motor_current = self.__act_motor_current
            return motor_current

    """
    Set function
    """

    def set_velocity_ratio(self, ratio):
        msg = Float64()
        msg.data = ratio
        self.__set_velocity_ratio_pub.publish(msg)

    def set_acceleration_ratio(self, ratio):
        msg = Float64()
        msg.data = ratio
        self.__set_acceleration_ratio_pub.publish(msg)

    def set_pose(self, pos, rot, wait_callback=True):
        assert not np.isnan(np.sum(pos))
        assert not np.isnan(np.sum(rot))
        joint = np.squeeze(dvrkKinematics.pose_to_joint(pos, rot))  # SAM: sometimes need to squeeze to avoid ROS error
        return self.set_joint(joint, wait_callback=wait_callback)

    def set_pose_direct(self, pos, rot):
        joint = dvrkKinematics.pose_to_joint(pos, rot)
        self.set_joint_direct(joint)

    # pose interpolation is based on 'cubic spline'
    def set_pose_interpolate(self, pos, rot, tf_init=0.5, t_step=0.01, v_max=None, a_max=None):
        """

        :param pos: [x,y,z]
        :param rot: [qx,qy,qz,qw]
        :param tf_init: initial guess to be minimized
        :param t_step:
        :return:
        """
        assert not np.isnan(np.sum(pos))
        assert not np.isnan(np.sum(rot))

        # Define q0 and qf
        pos0, rot0 = self.get_current_pose(wait_callback=True)
        if len(pos) == 0:
            posf = pos0
        else:
            posf = pos
        if len(rot) == 0:
            rotf = rot0
        else:
            rotf = rot

        rot0 = U.quaternion_to_euler(rot0)
        rotf = U.quaternion_to_euler(rotf)
        pose0 = np.concatenate((pos0, rot0))
        posef = np.concatenate((posf, rotf))

        vel_limit = dvrkVar.v_max if v_max is None else v_max
        acc_limit = dvrkVar.a_max if v_max is None else a_max

        # Define trajectory
        if np.allclose(pose0, posef):
            return False
        else:
            _, traj = self.cubic_cartesian(
                pose0, posef, vel_limit=vel_limit, acc_limit=acc_limit, tf_init=tf_init, t_step=0.01
            )
            # Execute trajectory
            for q in traj:
                self.set_joint_direct(q)
                rospy.sleep(t_step)
            return True

    def set_joint_direct(self, joint):
        assert not np.isnan(np.sum(joint))
        msg = JointState()
        msg.position = joint
        self.__set_position_joint_pub.publish(msg)

    def set_joint(self, joint, wait_callback=True):
        assert not np.isnan(np.sum(joint))
        msg = JointState()
        msg.position = joint
        if wait_callback:
            self.__goal_reached_event.clear()
            self.__set_position_goal_joint_pub.publish(msg)
            # self.__set_position_joint_pub.publish(msg)
            return self.__goal_reached_event.wait(3)  # 20 seconds at most
        else:
            self.__set_position_goal_joint_pub.publish(msg)
            # self.__set_position_joint_pub.publish(msg)
            return True

    def set_joint_interpolate(self, joint, method="cubic", t_step=0.01):
        assert not np.isnan(np.sum(joint))
        # Define q0 and qf
        q0 = self.get_current_joint(wait_callback=True)
        if len(joint) == 0:
            qf = q0
        else:
            qf = joint

        if np.allclose(q0, qf):
            return False
        else:
            # Define trajectory
            if method == "cubic":
                t, traj = self.cubic(q0, qf, v_max=dvrkVar.v_max, a_max=dvrkVar.a_max, t_step=t_step)
            elif method == "LSPB":
                t, traj = self.LSPB(q0, qf, v_max=dvrkVar.v_max, a_max=dvrkVar.a_max, t_step=t_step)
            else:
                raise IndexError

            # Execute trajectory
            for q in traj:
                self.set_joint_direct(q)
                rospy.sleep(t_step)
            return True

    def set_jaw_direct(self, jaw):
        assert not np.isnan(np.sum(jaw))
        msg = JointState()
        msg.position = jaw
        self.__set_position_jaw_pub.publish(msg)
        return True

    def set_jaw(self, jaw, wait_callback=True):
        assert not np.isnan(np.sum(jaw))
        msg = JointState()
        msg.position = jaw
        if wait_callback:
            self.__goal_reached_event.clear()
            self.__set_position_goal_jaw_pub.publish(msg)
            self.__goal_reached_event.wait()  # 10 seconds at most
            return True
        else:
            self.__set_position_goal_jaw_pub.publish(msg)
            return True

    # linear interpolation
    # this function doesn't issue the "goal_reached" flag at the end
    # Not reliable to use with set_pose or set_joint
    def set_jaw_interpolate(self, jaw, t_step=0.01):
        assert not np.isnan(np.sum(jaw))

        # Define q0 and qf
        jaw0 = self.get_current_jaw(wait_callback=True)
        if len(jaw) == 0:
            jawf = jaw0
        else:
            jawf = jaw
        q0 = jaw0
        qf = jawf

        # Define trajectory
        if np.allclose(q0, qf):
            return False
        else:
            t, traj = self.Linear(q0, qf, v=[6.0], t_step=t_step)
            # Execute trajectory
            for q in traj:
                self.set_jaw_direct(q)
                rospy.sleep(t_step)
            return True

    def set_pose_trajectory(self, pose_traj):
        self.set_pose_trajectory(pose_traj[0])
        for pose in pose_traj:
            start = time.perf_counter()
            self.set_pose_direct(pose)
            end = time.perf_counter()
            delta = 0.01 - (end - start)
            if delta > 0:
                time.sleep(delta)

    def set_joint_trajectory(self, q_pos):
        self.set_joint_interpolate(q_pos[0])
        for joint in q_pos:
            start = time.perf_counter()
            self.set_joint_direct(joint=joint)
            end = time.perf_counter()
            delta = 0.01 - (end - start)
            if delta > 0:
                time.sleep(delta)

    def set_jaw_trajectory(self, jaw_traj):
        self.set_jaw_interpolate(jaw_traj[0])
        for jaw in jaw_traj:
            start = time.perf_counter()
            self.set_jaw_direct(jaw=jaw)
            end = time.perf_counter()
            delta = 0.01 - (end - start)
            if delta > 0:
                time.sleep(delta)

    """
    Conversion function
    """
    # Depreciated. All poses are converted from joint angles
    # # Matching coordinate of the robot base and the end effector
    # def set_rot_transform(self, q):
    #     qx, qy, qz, qw = q
    #     R1 = PyKDL.Rotation.Quaternion(qx,qy,qz,qw)
    #     R2 = PyKDL.Rotation.EulerZYX(-np.pi/2, 0, 0)  # rotate -90 (deg) around z-axis
    #     R3 = PyKDL.Rotation.EulerZYX(0, np.pi, 0)  # rotate 180 (deg) around y-axis
    #     R = R1 * R2 * R3
    #     return R.GetQuaternion()
    #
    # # Matching coordinate of the robot base and the end effector
    # def get_rot_transform(self, q):
    #     qx, qy, qz, qw = q
    #     R1 = PyKDL.Rotation.Quaternion(qx, qy, qz, qw)
    #     R2 = PyKDL.Rotation.EulerZYX(0, np.pi, 0)  # rotate 180 (deg) around y-axis
    #     R3 = PyKDL.Rotation.EulerZYX(-np.pi/2, 0, 0)  # rotate -90 (deg) around z-axis
    #     R = R1 * R2.Inverse() * R3.Inverse()
    #     return R.GetQuaternion()

    """
    Trajectory
    """

    # q = [q1, ..., q6]
    def Linear(self, q0, qf, v, t_step):
        num_axis = len(q0)
        q0 = np.array(q0)
        qf = np.array(qf)
        v = np.array(v)

        if np.allclose(q0, qf):
            t = [0.0]
            joint = [qf]
            return t, joint

        # Design variables
        tf = abs((qf - q0) / v)  # total time taken

        # Calculate trajectories
        t = np.arange(start=0.0, stop=tf, step=t_step)
        joint = []
        for i in range(num_axis):
            # joint traj.
            q = (qf[i] - q0[i]) / tf[i] * t + q0[i]
            joint.append(q)
        joint = np.array(joint).T
        assert ~np.isnan(t).any()
        assert ~np.isnan(joint).any()
        return t, joint

    @classmethod
    # q = [q1, ..., q6]
    def LSPB(cls, q0, qf, v_max, a_max, t_step=0.01):
        q0 = np.array(q0)
        qf = np.array(qf)
        v_max = np.array(v_max)
        a_max = np.array(a_max)
        if np.allclose(q0, qf):
            t = [0.0]
            joint = [qf]
            return t, joint

        # Design variables
        A = max(abs((qf - q0) / a_max))
        B = max(abs((qf - q0) / v_max))
        tb = A / B
        tf = B + tb
        if tf < 2 * tb:
            tb = np.sqrt(A)
            tf = 2 * tb

        # Define coefficients
        A = np.array(
            [
                [tb**2, -tb, -1, 0.0, 0.0, 0.0],
                [2 * tb, -1, 0.0, 0.0, 0.0, 0.0],
                [0.0, tf - tb, 1, -((tf - tb) ** 2), -(tf - tb), -1],
                [0.0, 1.0, 0.0, -2 * (tf - tb), -1, 0.0],
                [0.0, 0.0, 0.0, 2 * tf, 1.0, 0.0],
                [0.0, 0.0, 0.0, tf**2, tf, 1.0],
            ]
        )
        b = np.block([[-q0], [np.zeros_like(q0)], [np.zeros_like(q0)], [np.zeros_like(q0)], [np.zeros_like(q0)], [qf]])
        coeff = np.linalg.inv(A).dot(b)
        a1 = coeff[0]
        a2 = coeff[1]
        b2 = coeff[2]
        a3 = coeff[3]
        b3 = coeff[4]
        c3 = coeff[5]

        # Calculate trajectories
        t = np.arange(start=0.0, stop=tf, step=t_step)
        t1 = t[t < tb].reshape(-1, 1)
        t2 = t[(tb <= t) & (t < tf - tb)].reshape(-1, 1)
        t3 = t[tf - tb <= t].reshape(-1, 1)

        # joint traj.
        traj1 = a1 * t1**2 + q0
        traj2 = a2 * t2 + b2
        traj3 = a3 * t3**2 + b3 * t3 + c3
        q_pos = np.concatenate((traj1, traj2, traj3))
        assert ~np.isnan(t).any()
        assert ~np.isnan(q_pos).any()
        return t, q_pos

    @classmethod
    # q = [q1, ..., q6]
    def cubic(cls, q0, qf, v_max, a_max, t_step=0.01):  # assume that v0 and vf are zero
        q0 = np.array(q0)
        qf = np.array(qf)
        v_max = np.array(v_max)
        a_max = np.array(a_max)
        if np.allclose(q0, qf):
            t = [0.0]
            joint = [qf]
            return t, joint

        # v_max = 1.5*(qf-q0)/tf
        tf_vel = 1.5 * (qf - q0) / v_max

        # a_max = 6*(qf-q0)/(tf**2)
        tf_acc = np.sqrt(abs(6 * (qf - q0) / a_max))
        tf_Rn = np.maximum(tf_vel, tf_acc)  # tf for each axis (nx1 array)
        tf = max(tf_Rn)  # maximum scalar value among axes

        # Define coefficients
        a = -2 * (qf - q0) / (tf**3)
        b = 3 * (qf - q0) / (tf**2)
        c = np.zeros_like(a)
        d = q0

        # Calculate trajectorie
        t = np.arange(start=0.0, stop=tf, step=t_step).reshape(-1, 1)

        # joint traj.
        q_pos = a * t**3 + b * t**2 + c * t + d
        assert ~np.isnan(t).any()
        assert ~np.isnan(q_pos).any()
        return t, q_pos

    @classmethod
    def __cubic_time(cls, pose0, posef, tf, t_step):
        pose0 = np.array(pose0)
        posef = np.array(posef)
        if np.allclose(pose0, posef):
            t = np.array([0.0])
            pose_traj = [posef]
            return t, pose_traj

        # Define coefficients
        a = -2 * (posef - pose0) / (tf**3)
        b = 3 * (posef - pose0) / (tf**2)
        c = np.zeros_like(a)
        d = pose0

        # Calculate trajectorie
        t = np.arange(start=0.0, stop=tf, step=t_step).reshape(-1, 1)
        pose_traj = a * t**3 + b * t**2 + c * t + d
        assert ~np.isnan(t).any()
        assert ~np.isnan(pose_traj).any()
        return t, pose_traj

    # pose = [x,y,z, yaw, pitch, roll]
    @classmethod
    def cubic_cartesian(cls, pose0, posef, vel_limit, acc_limit, tf_init=0.5, t_step=0.01):
        tf = tf_init
        while True:
            dt = 0.01
            _, pose_traj = dvrkArm.__cubic_time(pose0=pose0, posef=posef, tf=tf, t_step=dt)

            # pose_traj in Cartesian, q_pos in Joint space
            q_pos = dvrkKinematics.pose_to_joint(pos=pose_traj[:, :3], rot=U.euler_to_quaternion(pose_traj[:, 3:]))
            q_pos_prev = np.insert(q_pos, 0, q_pos[0], axis=0)
            q_pos_prev = np.delete(q_pos_prev, -1, axis=0)
            q_vel = (q_pos - q_pos_prev) / dt
            q_vel_prev = np.insert(q_vel, 0, q_vel[0], axis=0)
            q_vel_prev = np.delete(q_vel_prev, -1, axis=0)
            q_acc = (q_vel - q_vel_prev) / dt

            # find maximum values
            vel_max = np.max(abs(q_vel), axis=0)
            acc_max = np.max(abs(q_acc), axis=0)

            if np.any(vel_max > vel_limit) or np.any(acc_max > acc_limit):
                if tf_init == tf:
                    tf_init += 0.5
                    tf = tf_init
                else:
                    tf += 0.02
                    break
            else:
                if tf < 0.1:
                    break
                tf -= 0.01
        _, pose_traj = dvrkArm.__cubic_time(pose0=pose0, posef=posef, tf=tf, t_step=t_step)
        q_pos = dvrkKinematics.pose_to_joint(pos=pose_traj[:, :3], rot=U.euler_to_quaternion(pose_traj[:, 3:]))
        return tf, q_pos


if __name__ == "__main__":
    import time

    arm1 = dvrkArm("/PSM1")
    arm2 = dvrkArm("/PSM2")
    # traj = np.load("../new_traj.npy")
    # st = 0.0
    # arm1.set_joint(traj[0], wait_callback=True)
    # for q in traj:
    #     print(time.perf_counter() - st)
    #     st = time.perf_counter()
    #     arm1.set_joint_direct(q)
    #     time.sleep(0.01)

    pos1 = [0.005, 0.0, -0.13]
    rot1 = [0.0, 0.0, 0.0]
    pos2 = [-0.005, 0.0, -0.13]
    rot2 = [0.0, 0.0, 0.0]
    q1 = U.euler_to_quaternion(rot1, unit="deg")
    q2 = U.euler_to_quaternion(rot2, unit="deg")
    while True:
        arm2.set_pose(pos1, q1)
        arm2.set_pose(pos2, q2)

    # # jaw1 = [30 * np.pi / 180.]
    # # pos2 = [-0.12, 0.0, -0.13]
    # # rot2 = [0.0, 0.0, 0.0]
    # # q2 = U.euler_to_quaternion(rot2, unit='deg')
    # # jaw2 = [0.0]
    joint1 = [0.4, 0.0, 0.15, 0.0, 1.0, 0.0]
    joint2 = [0.4, 0.0, 0.15, 0.0, -1.0, 0.0]
    # q5 = joint1[4]
    # q6 = (0.830634273)/(1.01857984)*q5
    # joint1[-1] = q6
    # q5 = joint2[4]
    # q6 = (0.830634273)/(1.01857984)*q5
    # joint2[-1] = q6

    # # p1.set_joint(joint=joint1)
    # # p1.set_joint(joint=joint2)
    print("started")
    from scipy.spatial.transform import Rotation

    joint = [0.0, 0.0, 0.15, 0.0, 0.0, 0.5]
    pos, rot = dvrkKinematics.joint_to_pose(joint)
    RR = U.quaternion_to_R(rot)
    print(RR)
    while True:
        pass
        # time.sleep(1)
        # arm1.set_joint(joint=joint2, wait_callback=True)
        # print ("cleared")
        # arm1.set_joint(joint=joint2, wait_callback=False)
        # print ("joint 2 cleared")
        # arm1.set_pose_interpolate(pos=pos1, rot=q1)
        # arm1.set_pose_interpolate(pos=pos2, rot=q2)
    #     # arm1.set_joint_dinterpolate(joint=joint1, method='LSPB')
    #     # arm1.set_joint_interpolate(joint=joint2, method='LSPB')
    #     # print ("moved")
