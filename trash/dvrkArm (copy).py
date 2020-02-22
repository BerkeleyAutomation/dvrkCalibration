import threading
import rospy

import FLSpegtransfer.utils.CmnUtil as U
import numpy as np
import PyKDL

from tf_conversions import posemath
from std_msgs.msg import String, Bool, Float32, Empty, Float64MultiArray
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from sensor_msgs.msg import JointState


class dvrkArm(object):
    """Simple arm API wrapping around ROS messages
    """
    def __init__(self, arm_name, ros_namespace='/dvrk'):

        # data members, event based
        self.__arm_name = arm_name
        self.__ros_namespace = ros_namespace
        self.__goal_reached_event = threading.Event()
        self.__get_position_event = threading.Event()
        self.__get_velocity_event = threading.Event()
        self.__get_jaw = False
        self.__get_jaw_event = threading.Event()

        # continuous publish from dvrk_bridge
        self.__position_cartesian_current = PyKDL.Frame()
        self.__position_joint_current = np.array(0, dtype = np.float)
        self.__position_jaw_current = 0.0
        self.__velocity_cartesian_current = []

        self.__sub_list = []
        self.__pub_list = []

        # publisher
        frame = PyKDL.Frame()
        self.__full_ros_namespace = self.__ros_namespace + self.__arm_name
        self.__set_position_joint_pub = rospy.Publisher(self.__full_ros_namespace + '/set_position_joint', JointState,
                                                        latch = True, queue_size = 1)
        self.__set_position_goal_joint_pub = rospy.Publisher(self.__full_ros_namespace + '/set_position_goal_joint',
                                                             JointState, latch = True, queue_size = 1)
        self.__set_position_cartesian_pub = rospy.Publisher(self.__full_ros_namespace
                                                            + '/set_position_cartesian',
                                                            Pose, latch = True, queue_size = 1)
        self.__set_position_goal_cartesian_pub = rospy.Publisher(self.__full_ros_namespace
                                                                 + '/set_position_goal_cartesian',
                                                                 Pose, latch = True, queue_size = 1)
        self.__set_position_jaw_pub = rospy.Publisher(self.__full_ros_namespace
                                                      + '/set_position_jaw',
                                                      JointState, latch = True, queue_size = 1)
        self.__set_position_goal_jaw_pub = rospy.Publisher(self.__full_ros_namespace
                                                           + '/set_position_goal_jaw',
                                                           JointState, latch = True, queue_size = 1)

        self.__pub_list = [self.__set_position_joint_pub,
                           self.__set_position_goal_joint_pub,
                           self.__set_position_cartesian_pub,
                           self.__set_position_goal_cartesian_pub,
                           self.__set_position_jaw_pub,
                           self.__set_position_goal_jaw_pub]

        self.__sub_list = [rospy.Subscriber(self.__full_ros_namespace + '/goal_reached',
                                          Bool, self.__goal_reached_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/position_cartesian_current',
                                          PoseStamped, self.__position_cartesian_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/io/joint_position',
                                            JointState, self.__position_joint_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/state_jaw_current',
                                            JointState, self.__position_jaw_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/twist_body_current',
                                            TwistStamped, self.__velocity_cartesian_current_cb)]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('dvrkArm_node', anonymous = True, log_level = rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')
        self.interval_ms = 30  # Sept 6: Minho suggests 20ms --> 30ms?
        self.rate = rospy.Rate(1000.0 / self.interval_ms)

    """
    Callback function
    """
    def __goal_reached_cb(self, data):
        self.__goal_reached = data.data
        self.__goal_reached_event.set()

    def __position_cartesian_current_cb(self, data):
        self.__position_cartesian_current = posemath.fromMsg(data.pose)
        self.__get_position_event.set()

    def __position_joint_current_cb(self, data):
        self.__position_joint_current.resize(len(data.position))
        self.__position_joint_current.flat[:] = data.position

    def __position_jaw_current_cb(self, data):
        self.__position_jaw_current = data.position
        self.__get_jaw = True
        self.__get_jaw_event.set()

    def __velocity_cartesian_current_cb(self, data):
        self.__velocity_cartesian_current = [data.twist.linear.x, data.twist.linear.y, data.twist.linear.z, data.twist.angular.x, data.twist.angular.y, data.twist.angular.z]
        self.__get_velocity_event.set()

    """
    Get States function
    """
    def get_current_pose_frame(self):
        """

        :return: PyKDL.Frame
        """
        return self.__position_cartesian_current

    def get_current_orientation_quaternion(self):
        return self.__position_cartesian_current.M.GetQuaternion()

    def get_current_position(self, wait_callback=False):
        if wait_callback:
            self.__get_position_event.clear()
            if self.__get_position_event.wait(20):  # 1 minute at most
                return list(self.__position_cartesian_current.p)
            else:
                return []
        else:
            return list(self.__position_cartesian_current.p)

    def get_current_pose(self,unit='rad'):    # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        raise NotImplementedError
        pos,rot = self.PyKDLFrame_to_NumpyArray(self.__position_cartesian_current)
        if unit == 'deg':
            rot = U.rad_to_deg(rot)
        return pos,rot

    def get_current_pose_and_wait(self, unit='rad'):    # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        raise NotImplementedError
        self.__get_position_event.clear()

        # the position is originally not received
        # self.__get_position = False
        # recursively call this function until the position is received
        # self.__get_position_event.wait(20)  # 1 minute at most

        if self.__get_position_event.wait(20):  # 1 minute at most
            pos, rot = self.PyKDLFrame_to_NumpyArray(self.__position_cartesian_current)
            if unit == 'deg':
                rot = U.rad_to_deg(rot)
            return pos, rot
        else:
            return [], []

    def get_current_joint(self, unit='rad'):
        """

        :param unit: 'rad' or 'deg'
        :return: List
        """
        joint = self.__position_joint_current
        print joint
        if unit == 'deg':
            joint = U.rad_to_deg(self.__position_joint_current)
            joint[2] = self.__position_joint_current[2]
        return joint

    def get_current_jaw(self,unit='rad'):
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.float64
        """
        jaw = np.float64(self.__position_jaw_current)
        if unit == "deg":
            jaw = U.rad_to_deg(self.__position_jaw_current)
        return jaw

    def get_current_jaw_and_wait(self, unit='rad'):    # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        self.__get_jaw_event.clear()

        # the position is originally not received
        self.__get_jaw = False
        # recursively call this function until the position is received
        self.__get_jaw_event.wait(20)  # 1 minute at most

        if self.__get_jaw:
            jaw = np.float64(self.__position_jaw_current)
            if unit == "deg":
                jaw = U.rad_to_deg(self.__position_jaw_current)
            return jaw
        else:
            return []

    def get_current_velocity(self, wait_callback=False):
        if wait_callback:
            self.__get_velocity_event.clear()
            if self.__get_velocity_event.wait(20):  # 1 minute at most
                return self.__velocity_cartesian_current
            else:
                return []
        else:
            return self.__velocity_cartesian_current

    """
    Set States function
    """
    def set_pose_frame(self, frame):
        """

        :param frame: PyKDL.Frame
        """
        msg = posemath.toMsg(frame)
        return self.__set_position_goal_cartesian_publish_and_wait(msg)

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

    def set_pose(self, pos, rot, unit='rad', wait_callback=True):
        """

        :param pos_des: position array [x,y,z]
        :param rot_des: rotation array [Z,Y,X euler angle]
        :param unit: 'rad' or 'deg'
        :param wait_callback: True or False
        """
        if unit == 'deg':
            rot = U.deg_to_rad(rot)

        # limiting range of motion
        limit_angle = 89 * np.pi / 180
        if rot[1] > limit_angle:      rot[1] = limit_angle
        elif rot[1] < -limit_angle:   rot[1] = -limit_angle
        if rot[2] > limit_angle:      rot[2] = limit_angle
        elif rot[2] < -limit_angle:   rot[2] = -limit_angle

        # set in position cartesian mode
        frame = self.NumpyArraytoPyKDLFrame(pos, rot)
        msg = posemath.toMsg(frame)

        print msg
        # go to that position by goal
        if wait_callback:
            return self.__set_position_goal_cartesian_publish_and_wait(msg)
        else:
            self.goal_reached = False
            self.__set_position_goal_cartesian_pub.publish(msg)
            return True

    def set_pose_direct(self, pos, rot, unit='rad'):
        """

        :param pos_des: position array [x,y,z]
        :param rot_des: rotation array [Z,Y,X euler angle]
        :param unit: 'rad' or 'deg'
        """
        if unit == 'deg':
            rot = U.deg_to_rad(rot)

        # limiting range of motion
        limit_angle = 89*np.pi/180
        if rot[1] > limit_angle:      rot[1] = limit_angle
        elif rot[1] < -limit_angle:   rot[1] = -limit_angle
        if rot[2] > limit_angle:      rot[2] = limit_angle
        elif rot[2] < -limit_angle:   rot[2] = -limit_angle

        # set in position cartesian mode
        frame = self.NumpyArraytoPyKDLFrame(pos, rot)
        msg = posemath.toMsg(frame)
        # go to that position by goal
        self.__set_position_cartesian_pub.publish(msg)

    # specify intermediate points between q0 & qf using linear interpolation (blocked until goal reached)
    def set_pose_linear(self, pos, rot, unit='rad'):
        q0 = self.get_current_position(wait_callback=True)
        qf = pos
        assert len(qf) > 0, qf
        assert len(q0) > 0, q0
        
        if np.allclose(q0,qf):
            return False
        else:
            tf = np.linalg.norm(np.array(qf)-np.array(q0))**0.8 * 10
            v_limit = (np.array(qf)-np.array(q0))/tf
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

    def __set_position_goal_cartesian_publish_and_wait(self, msg):
        """

        :param msg: pose
        :returns: returns true if the goal is reached
        """
        self.__goal_reached_event.clear()
        # the goal is originally not reached
        self.__goal_reached = False
        # recursively call this function until end is reached
        self.__set_position_goal_cartesian_pub.publish(msg)
        self.__goal_reached_event.wait(20) # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

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
        self.__goal_reached_event.wait(20) # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

    def set_jaw(self, jaw, unit='rad', wait_callback=True):
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
            self.__set_position_goal_jaw_pub.publish(msg)
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
        if np.allclose(q0,qf):
            return False
        else:
            tf = np.linalg.norm(np.array(qf)-np.array(q0))**0.8 * 0.6
            v_limit = (np.array(qf)-np.array(q0))/tf
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

    def __set_position_goal_jaw_publish_and_wait(self,msg):
        """

        :param msg:
        :return: whether or not you have successfully moved by goal or not
        """
        self.__goal_reached_event.clear()
        # the goal is originally not reached
        self.__goal_reached = False
        # recursively call this function until end is reached
        self.__set_position_goal_jaw_pub.publish(msg)
        self.__goal_reached_event.wait(20) # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

    """
    Conversion function
    """
    def PyKDLFrame_to_NumpyArray(self,frame):
        raise NotImplementedError
        pos = np.array([frame.p[0], frame.p[1], frame.p[2]])
        rz, ry, rx = frame.M.GetEulerZYX()
        if rx >= 0:     rot  = np.array([-rz,ry,-rx]) + np.array([np.pi/2, 0, np.pi])
        elif rx < 0:    rot  = np.array([-rz,ry,-rx]) + np.array([np.pi/2, 0, -np.pi])
        return pos,rot

    def NumpyArraytoPyKDLFrame(self,pos,rot, unit='rad'):
        if unit == 'deg':
            rot = U.deg_to_rad(rot)
        px, py, pz = pos
        rz, ry, rx = np.array([-np.pi/2, np.pi, 0]) - np.array(rot)
        return PyKDL.Frame(PyKDL.Rotation.EulerZYX(rz, ry, rx), PyKDL.Vector(px, py, pz))

    def ZYXtoquaternion(self, rot, unit='rad'):
        if unit == 'deg':
            rot = U.deg_to_rad(rot)
        rz, ry, rx = np.array([-np.pi / 2, np.pi, 0]) - np.array(rot)
        R = PyKDL.Rotation.EulerZYX(rz, ry, rx)
        return R.GetQuaternion()

    def NumpyArraytoPyKDLFrame_quaternion(self,pos,rot):
        px, py, pz = pos
        rx, ry, rz, rw = rot
        return PyKDL.Frame(PyKDL.Rotation.Quaternion(rx, ry, rz, rw), PyKDL.Vector(px, py, pz))

    """
    Trajectory
    """
    def LSPB(self, q0, qf, t, tf, v):

        if np.allclose(q0,qf):    return q0
        elif np.all(v)==0:    return q0
        elif tf==0:     return q0
        elif tf<0:     return []
        elif t<0:      return []
        else:
            v_limit = (np.array(qf) - np.array(q0)) / tf
            if np.allclose(U.normalize(v),U.normalize(v_limit)):
                if np.linalg.norm(v) < np.linalg.norm(v_limit) or np.linalg.norm(2*v_limit) < np.linalg.norm(v):
                    return []
                else:
                    tb = np.linalg.norm(np.array(q0)-np.array(qf)+np.array(v)*tf) / np.linalg.norm(v)
                    a = np.array(v)/tb
                    if 0 <= t and t < tb:
                        q = np.array(q0) + np.array(a)/2*t*t
                    elif tb < t and t <= tf - tb:
                        q = (np.array(qf)+np.array(q0)-np.array(v)*tf)/2 + np.array(v)*t
                    elif tf - tb < t and t <= tf:
                        q = np.array(qf)-np.array(a)*tf*tf/2 + np.array(a)*tf*t - np.array(a)/2*t*t
                    else:
                        return []
                    return q
            else:
                return []

if __name__ == "__main__":
    p1 = dvrkArm('/PSM1')
    p2 = dvrkArm('/PSM2')
    # print p2.get_current_position(wait_callback=True)
    # print p2.get_current_position()
    #
    # pos_org1 = [0.030, -0.02, -0.11]  # xyz position in (m)
    # rot_org1 = [30.0, 0.0, 0.0]  # ZYX Euler angle in (deg)
    # jaw_org1 = [0.0]  # jaw angle in (deg)
    #
    # pos_org2 = [-0.030, -0.02, -0.11]  # xyz position in (m)
    # rot_org2 = [-90.0, 0.0, 0.0]  # ZYX Euler angle in (deg)
    # jaw_org2 = [0.0]  # jaw angle in (deg)
    #
    # pos_org3 = [-0.030, -0.02, -0.11]  # xyz position in (m)
    # rot_org3 = [0.0, 0.0, 0.0]  # ZYX Euler angle in (deg)
    # jaw_org3 = [0.0]  # jaw angle in (deg)
    #
    pos_org1 = [0.1152513, - 0.06515163, - 0.15]
    rot_org1 = [-90.0, 0, 0]
    # pos_org2 = [0.0645287, - 0.06434801, - 0.15]
    # rot_org2 = [-90.0, 0, 0]
    #
    p1.set_pose(pos_org1, rot_org1, 'deg')
    # # print p1.get_current_p ose('deg')
    # p1.set_pose(pos_org2, rot_org2, 'deg')
