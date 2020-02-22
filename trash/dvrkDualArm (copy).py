import threading
import rospy

import dvrk.utils.CmnUtil as U
import numpy as np
import PyKDL

from tf_conversions import posemath
from std_msgs.msg import String, Bool, Float32, Empty, Float64MultiArray
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

class dvrkDualArm(object):
    """Dual arm API wrapping around ROS messages
    """
    def __init__(self, ros_namespace='/dvrk'):

        # data members, event based
        self.__arm_name = ['/PSM1', '/PSM2']
        self.__ros_namespace = ros_namespace
        self.__get_position_event = [threading.Event(), threading.Event()]
        self.__get_jaw_event = [threading.Event(), threading.Event()]

        # continuous publish from dvrk_bridge
        self.__position_cartesian_current_msg = [Pose(), Pose()]
        self.__position_cartesian_current_frame = [PyKDL.Frame(), PyKDL.Frame()]
        self.__position_joint_current = [0.0, 0.0]
        self.__position_jaw_current = [0.0, 0.0]
        self.__pose_cartesian_current = [0.0, 0.0]

        self.__sub_list = []
        self.__pub_list = []

        # publisher
        frame = PyKDL.Frame()
        self.__full_ros_namespace = [self.__ros_namespace + name for name in self.__arm_name]
        self.__set_position_joint_pub = [rospy.Publisher(name + '/set_position_joint', JointState,
                                                        latch=True, queue_size=1), name in self.__full_ros_namespace]
        self.__set_position_goal_joint_pub = [rospy.Publisher(name + '/set_position_goal_joint',
                                                             JointState, latch=True, queue_size=1) for name in self.__full_ros_namespace]
        self.__set_position_cartesian_pub = [rospy.Publisher(name + '/set_position_cartesian', Pose, latch=True, queue_size=1) for name in self.__full_ros_namespace]
        self.__set_position_goal_cartesian_pub = [rospy.Publisher(name + '/set_position_goal_cartesian', Pose, latch=True, queue_size=1) for name in self.__full_ros_namespace]
        self.__set_position_jaw_pub = [rospy.Publisher(name + '/set_position_jaw', JointState, latch=True, queue_size=1) for name in self.__full_ros_namespace]
        self.__set_position_goal_jaw_pub = [rospy.Publisher(name + '/set_position_goal_jaw', JointState, latch=True, queue_size=1) for name in self.__full_ros_namespace]

        self.__pub_list = [self.__set_position_joint_pub,
                           self.__set_position_goal_joint_pub,
                           self.__set_position_cartesian_pub,
                           self.__set_position_goal_cartesian_pub,
                           self.__set_position_jaw_pub,
                           self.__set_position_goal_jaw_pub]

        self.__position_cartesian_current_sub = [rospy.Subscriber(name + '/position_cartesian_current',
                                            PoseStamped, self.__position_cartesian_current_cb) for name in self.__full_ros_namespace]
        self.__position_joint_current_sub = [rospy.Subscriber(self.__full_ros_namespace[0] + '/state_joint_current',
                                            JointState, self.__position_joint_current1_cb),
                                             rospy.Subscriber(self.__full_ros_namespace[1] + '/state_joint_current',
                                                              JointState, self.__position_joint_current2_cb)]
        self.__position_jaw_current_sub = [rospy.Subscriber(self.__full_ros_namespace[0] + '/state_jaw_current',
                                            JointState, self.__position_jaw_current1_cb),
                                            rospy.Subscriber(self.__full_ros_namespace[1] + '/state_jaw_current',
                                            JointState, self.__position_jaw_current2_cb)]

        self.__sub_list = [self.__position_cartesian_current_sub,
                           self.__position_joint_current_sub,
                           self.__position_jaw_current_sub]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('dvrkDualArm_node', anonymous=True, log_level=rospy.WARN)
            self.interval_ms = 20
            self.rate = rospy.Rate(1000.0 / self.interval_ms)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

    """
    Callback function
    """

    def __position_cartesian_current_cb(self, data):
        """Callback for the current cartesian position.
        """
        if data.header.frame_id == "PSM1":
            self.__position_cartesian_current_msg[0] = data.pose
            self.__position_cartesian_current_frame[0] = posemath.fromMsg(data.pose)
            self.__get_position_event[0].set()
        elif data.header.frame_id == "PSM2":
            self.__position_cartesian_current_msg[1] = data.pose
            self.__position_cartesian_current_frame[1] = posemath.fromMsg(data.pose)
            self.__get_position_event[1].set()
        self.__pose_cartesian_current = self.get_current_pose('rad')

    def __position_joint_current1_cb(self, data):
        """Callback for the current joint position.
        """
        self.__position_joint_current[0] = list(data.position)

    def __position_joint_current2_cb(self, data):
        """Callback for the current joint position.
        """
        self.__position_joint_current[1] = list(data.position)

    def __position_jaw_current1_cb(self, data):
        """Callback for the current jaw position.
        """
        self.__position_jaw_current[0] = data.position[0]
        self.__get_jaw_event[0].set()

    def __position_jaw_current2_cb(self, data):
        """Callback for the current jaw position.
        """
        self.__position_jaw_current[1] = data.position[0]
        self.__get_jaw_event[1].set()

    """
    Get States function
    """

    def get_current_pose(self, unit='rad'):  # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        pos1, rot1 = self.PyKDLFrame2List(self.__position_cartesian_current_frame[0])
        pos2, rot2 = self.PyKDLFrame2List(self.__position_cartesian_current_frame[1])
        jaw1 = [self.__position_jaw_current[0]]
        jaw2 = [self.__position_jaw_current[1]]
        if unit == 'deg':
            rot1 = U.rad_to_deg(rot1)
            rot2 = U.rad_to_deg(rot2)
            jaw1 = U.rad_to_deg(jaw1)
            jaw2 = U.rad_to_deg(jaw2)
        pose1 = pos1+rot1+jaw1
        pose2 = pos2+rot2+jaw2
        return [pose1, pose2]

    def get_current_pose_and_wait(self, unit='rad'):  # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        self.__get_position_event[0].clear()
        self.__get_position_event[1].clear()
        self.__get_jaw_event[0].clear()
        self.__get_jaw_event[1].clear()
        if self.__get_position_event[0].wait(20) and self.__get_position_event[1].wait(20)\
            and self.__get_jaw_event[0].wait(20) and self.__get_jaw_event[1].wait(20):
            pos1, rot1 = self.PyKDLFrame_to_NumpyArray(self.__position_cartesian_current_frame[0])
            pos2, rot2 = self.PyKDLFrame_to_NumpyArray(self.__position_cartesian_current_frame[1])
            jaw1 = self.__position_jaw_current[0]
            jaw2 = self.__position_jaw_current[1]
            if unit == 'deg':
                rot1 = U.rad_to_deg(rot1)
                rot2 = U.rad_to_deg(rot2)
                jaw1 = U.rad_to_deg(jaw1)
                jaw2 = U.rad_to_deg(jaw2)

            pose1 = np.array([pos1, rot1, jaw1]).flatten
            pose2 = np.array([pos2, rot2, jaw2]).flatten
            return np.array([pose1, pose2])
        else:
            return [], []

    def get_current_joint(self, unit='rad'):
        """

        :param unit: 'rad' or 'deg'
        :return: List
        """
        joint = self.__position_joint_current
        if unit == 'deg':
            joint = U.rad_to_deg(self.__position_joint_current)
            joint[2] = self.__position_joint_current[2]
        return joint

    def get_current_jaw(self, unit='rad'):
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.float64
        """
        jaw = np.float64(self.__position_jaw_current)
        if unit == "deg":
            jaw = U.rad_to_deg(self.__position_jaw_current)
        return jaw

    def get_current_jaw_and_wait(self, unit='rad'):  # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        self.__get_jaw_event[0].clear()
        self.__get_jaw_event[1].clear()
        if self.__get_jaw_event[0].wait(20) and self.__get_jaw_event[1].wait(20):
            jaw = np.float64(self.__position_jaw_current)
            if unit == "deg":
                jaw = U.rad_to_deg(self.__position_jaw_current)
            return jaw
        else:
            return []

    """
    Set States function
    """

    def set_pose_frame(self, frame1, frame2):
        """

        :param frame: PyKDL.Frame
        """
        msg1 = posemath.toMsg(frame1)
        msg2 = posemath.toMsg(frame2)
        self.__set_position_goal_cartesian_publish_and_wait(msg1)
        self.__set_position_goal_cartesian_publish_and_wait(msg2)
        return True

    def set_pose_quaternion(self, pos, rot, wait_callback=True):
        """

        :param pos: position array [x,y,z]
        :param rot: orientation array in quaternion [x,y,z,w]
        :param wait_callback: True or False
        """
        # set in position cartesian mode
        frame = self.NumpyArraytoPyKDLFrame_quaternion(pos, rot)
        msg = posemath.toMsg(frame)
        # go to that position by goal
        if wait_callback:
            return self.__set_position_goal_cartesian_publish_and_wait(msg)
        else:
            self.__set_position_goal_cartesian_pub.publish(msg)
            return True

    def set_pose(self, pose, unit='rad', wait_callback=True):
        """

        :param pose: [[pos1, rot1, jaw1],[pos2, rot2, jaw2]]
               pos1, pos2: position array [x,y,z]
               rot1, rot2: rotation array [Z,Y,X] in euler angle
               jaw1, jaw2: jaw angle array
        :param unit: 'rad' or 'deg'
        :param wait_callback: True or False
        """
        pos1 = pose[0][0:3]
        rot1 = pose[0][3:6]
        jaw1 = pose[0][6]
        pos2 = pose[1][0:3]
        rot2 = pose[1][3:6]
        jaw2 = pose[1][6]
        if unit == 'deg':
            rot1 = U.deg_to_rad(rot1)
            rot2 = U.deg_to_rad(rot2)
            jaw1 = U.deg_to_rad(jaw1)
            jaw2 = U.deg_to_rad(jaw2)
        # set in position cartesian mode
        frame1 = self.NumpyArraytoPyKDLFrame(pos1, rot1)
        frame2 = self.NumpyArraytoPyKDLFrame(pos2, rot2)
        msg1 = posemath.toMsg(frame1)
        msg2 = posemath.toMsg(frame2)
        # go to that position by goal
        if wait_callback:
            goal_reached = False
            while not goal_reached:
                self.rate.sleep()
                self.__set_position_goal_cartesian_pub[0].publish(msg1)
                self.__set_position_goal_cartesian_pub[1].publish(msg2)
                self.set_jaw(jaw1, unit, False)
                # print np.array(pose) - np.array(self.__pose_cartesian_current)
                if np.allclose(pose, self.__pose_cartesian_current):
                    print "True"
                    goal_reached = True
            return True
        else:
            self.__set_position_goal_cartesian_pub[0].publish(msg1)
            self.__set_position_goal_cartesian_pub[1].publish(msg1)
            return True

    def set_pose_direct(self, pos, rot, unit='rad'):
        """

        :param pos_des: position array [x,y,z]
        :param rot_des: rotation array [Z,Y,X euler angle]
        :param unit: 'rad' or 'deg'
        """
        if unit == 'deg':
            rot = U.deg_to_rad(rot)

        # set in position cartesian mode
        frame = self.NumpyArraytoPyKDLFrame(pos, rot)
        msg = posemath.toMsg(frame)
        # go to that position by goal
        self.__set_position_cartesian_pub.publish(msg)

    # specify intermediate points between q0 & qf using linear interpolation (blocked until goal reached)
    def set_pose_linear(self, pos, rot, unit='rad'):

        [q0, trash] = self.get_current_pose_and_wait()
        qf = pos
        assert len(qf) > 0, qf
        assert len(q0) > 0, q0

        if np.allclose(q0, qf):
            return False
        else:
            tf = np.linalg.norm(np.array(qf) - np.array(q0)) ** 0.8 * 10
            v_limit = (np.array(qf) - np.array(q0)) / tf
            v = v_limit * 1.5
            # print '\n'
            # print 'q0=', q0
            # print 'qf=', qf
            # print 'norm=', np.linalg.norm(np.array(qf) - np.array(q0))
            # print 'tf=', tf
            # print 'v=',v
            t = 0.0
            while True:
                q = self.LSPB(q0, qf, t, tf, v)
                # print q
                self.set_pose(q, rot, unit, False)
                # self.set_pose_direct(q, rot, unit)
                t += 0.001 * self.interval_ms
                self.rate.sleep()
                if t > tf:
                    break

    def set_joint(self, joint, unit='rad', wait_callback=True):
        """

        :param joint: joint array [j1, ..., j6]
        :param unit: 'rad', or 'deg'
        :param wait_callback: True or False
        """
        if unit == 'deg':
            joint = U.deg_to_rad(joint)
        msg = JointState()
        msg.position = joint
        if wait_callback:
            return self.__set_position_goal_joint_publish_and_wait(msg)
        else:
            self.__set_position_goal_joint_pub.publish(msg)
            return True

    def __set_position_goal_joint_publish_and_wait(self, msg):
        """

        :param msg: there is only one parameter, msg which tells us what the ending position is
        :returns: whether or not you have successfully moved by goal or not
        """
        self.__goal_reached_event.clear()
        # the goal is originally not reached
        self.__goal_reached = False
        # recursively call this function until end is reached
        self.__set_position_goal_joint_pub.publish(msg)
        self.__goal_reached_event.wait(20)  # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

    def set_jaw1(self, jaw, unit='rad', wait_callback=True):
        """

        :param jaw: jaw angle
        :param unit: 'rad' or 'deg'
        :param wait_callback: True or False
        """
        if unit == 'deg':
            jaw = U.deg_to_rad(jaw)
        msg = JointState()
        msg.position = [jaw]
        if wait_callback:
            return self.__set_position_goal_jaw_publish_and_wait(msg)
        else:
            self.__set_position_goal_jaw_pub[0].publish(msg)
            return True

    def set_jaw_direct(self, jaw, unit='rad'):
        """

        :param jaw: jaw angle
        :param unit: 'rad' or 'deg'
        """
        if unit == 'deg':
            jaw = U.deg_to_rad(jaw)
        msg = JointState()
        msg.position = [jaw]
        self.__set_position_jaw_pub.publish(msg)

    # specify intermediate points between q0 & qf using linear interpolation (blocked until goal reached)
    def set_jaw_linear(self, jaw, unit='rad'):

        q0 = self.get_current_jaw_and_wait()
        qf = jaw
        if np.allclose(q0, qf):
            return False
        else:
            tf = np.linalg.norm(np.array(qf) - np.array(q0)) ** 0.8 * 0.6
            v_limit = (np.array(qf) - np.array(q0)) / tf
            v = v_limit * 1.5
            t = 0.0
            while True:
                q = self.LSPB(q0, qf, t, tf, v)
                # print q
                # self.set_pose(q, rot, unit, False)
                self.set_jaw_direct(q, unit)
                t += 0.001 * self.interval_ms
                self.rate.sleep()
                if t > tf:
                    break

    def __set_position_goal_jaw_publish_and_wait(self, msg):
        """

        :param msg:
        :return: whether or not you have successfully moved by goal or not
        """
        self.__goal_reached_event.clear()
        # the goal is originally not reached
        self.__goal_reached = False
        # recursively call this function until end is reached
        self.__set_position_goal_jaw_pub.publish(msg)
        self.__goal_reached_event.wait(20)  # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

    """
    Conversion function
    """

    def PyKDLFrame2List(self, frame):
        pos = [frame.p[0], frame.p[1], frame.p[2]]
        rz, ry, rx = frame.M.GetEulerZYX()
        if rx >= 0:     rot = np.array([np.pi / 2, 0, np.pi]) - np.array([rz, ry, rx])
        elif rx < 0:    rot = np.array([np.pi / 2, 0, -np.pi]) - np.array([rz, ry, rx])
        return pos, list(rot)

    def NumpyArraytoPyKDLFrame(self, pos, rot):
        px, py, pz = pos
        rz, ry, rx = np.array([np.pi / 2, 0, -np.pi]) - np.array(rot)
        return PyKDL.Frame(PyKDL.Rotation.EulerZYX(rz, ry, rx), PyKDL.Vector(px, py, pz))

    def NumpyArraytoPyKDLFrame_quaternion(self, pos, rot):
        px, py, pz = pos
        rx, ry, rz, rw = rot
        return PyKDL.Frame(PyKDL.Rotation.Quaternion(rx, ry, rz, rw), PyKDL.Vector(px, py, pz))

    """
    Trajectory
    """

    def LSPB(self, q0, qf, t, tf, v):

        if np.allclose(q0, qf):
            return q0
        elif np.all(v) == 0:
            return q0
        elif tf == 0:
            return q0
        elif tf < 0:
            return []
        elif t < 0:
            return []
        else:
            v_limit = (np.array(qf) - np.array(q0)) / tf
            if np.allclose(U.normalize(v), U.normalize(v_limit)):
                if np.linalg.norm(v) < np.linalg.norm(v_limit) or np.linalg.norm(2 * v_limit) < np.linalg.norm(v):
                    return []
                else:
                    tb = np.linalg.norm(np.array(q0) - np.array(qf) + np.array(v) * tf) / np.linalg.norm(v)
                    a = np.array(v) / tb
                    if 0 <= t and t < tb:
                        q = np.array(q0) + np.array(a) / 2 * t * t
                    elif tb < t and t <= tf - tb:
                        q = (np.array(qf) + np.array(q0) - np.array(v) * tf) / 2 + np.array(v) * t
                    elif tf - tb < t and t <= tf:
                        q = np.array(qf) - np.array(a) * tf * tf / 2 + np.array(a) * tf * t - np.array(a) / 2 * t * t
                    else:
                        return []
                    return q
            else:
                return []


if __name__ == "__main__":
    p = dvrkDualArm()
    while True:
        pos1 = [-0.05, -0.05, -0.14]
        rot1 = [0,0,0]
        jaw1 = [0]
        pos2 = [-0.05, -0.05, -0.14]
        rot2 = [0,0,0]
        jaw2 = [0]
        pose = [pos1+rot1+jaw1, pos2+rot2+jaw2]
        p.set_pose(pose,'deg')

        pos1 = [0.05, 0.05, -0.14]
        rot1 = [0, 0, 0]
        jaw1 = [50]
        pos2 = [0.05, 0.05, -0.14]
        rot2 = [0, 0, 0]
        jaw2 = [50]
        pose = pose = [pos1+rot1+jaw1, pos2+rot2+jaw2]
        p.set_pose(pose, 'deg')

    # print p.get_current_pose()
    # print p.get_current_pose('deg')
