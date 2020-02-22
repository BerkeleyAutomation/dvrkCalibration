"""Shared methods, to be loaded in other code.
"""
import numpy as np

ESC_KEYS = [27, 1048603]
MILLION = float(10**6)

def rad_to_deg(rad):
    return np.array(rad) *180./np.pi

def deg_to_rad(deg):
    return np.array(deg) *np.pi/180.

def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

# def LPF(data_curr, data_prev, fc, dt):
#     return 2*np.pi*fc*dt*data_curr + (1-2*np.pi*fc*dt)*data_prev;

def LPF(raw_data, fc, dt):
    filtered = np.zeros_like(raw_data)
    for i in range(len(raw_data)):
        if i==0:
            filtered[0] = raw_data[0]
        else:
            filtered[i] = 2*np.pi*fc*dt*raw_data[i] + (1-2*np.pi*fc*dt)*filtered[i-1]
    return filtered

def euler_to_quaternion(rot, unit='rad'):
    if unit=='deg':
        rot = deg_to_rad(rot)

    # for the various angular functions
    z,y,x = rot
    cy = np.cos(z * 0.5);
    sy = np.sin(z * 0.5);
    cp = np.cos(y * 0.5);
    sp = np.sin(y * 0.5);
    cr = np.cos(x * 0.5);
    sr = np.sin(x * 0.5);

    # quaternion
    qw = cy * cp * cr + sy * sp * sr;
    qx = cy * cp * sr - sy * sp * cr;
    qy = sy * cp * sr + cy * sp * cr;
    qz = sy * cp * cr - cy * sp * sr;

    return [qx, qy, qz, qw]

def quaternion_to_eulerAngles(q, unit='rad'):
    qx, qy, qz, qw = q

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz);
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
    roll = np.arctan2(sinr_cosp, cosr_cosp);

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx);
    if (abs(sinp) >= 1):    pitch = np.sign(sinp)*(np.pi/2); # use 90 degrees if out of range
    else:                   pitch = np.arcsin(sinp);

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy);
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    yaw = np.arctan2(siny_cosp, cosy_cosp);

    if unit=='deg':
        [roll, pitch, yaw] = rad_to_deg([roll, pitch, yaw])
    return [roll,pitch,yaw]

def R_to_euler(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def R_to_quaternion(R):
    qw = np.sqrt((1. + R[0][0] + R[1][1] + R[2][2])/2.)
    qx = (R[2][1] - R[1][2]) / (4. * qw)
    qy = (R[0][2] - R[2][0]) / (4. * qw)
    qz = (R[1][0] - R[0][1]) / (4. * qw)
    return qx,qy,qz,qw

# quat_des = []
# for q in q_des:
#     R = BD.fk_orientation(q[0], q[1], q[2], q[3], q[4], q[5])
#     R_matrix = PyKDL.Rotation(R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2])
#     quat_des.append(list(R_matrix.GetQuaternion()))

# Get a rigid transformation matrix from pts1 to pts2
def get_rigid_transform(pts1, pts2):
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    mean1 = pts1.mean(axis=0)
    mean2 = pts2.mean(axis=0)
    pts1 = np.array([p - mean1 for p in pts1])
    pts2 = np.array([p - mean2 for p in pts2])
    # if option=='clouds':
    H = pts1.T.dot(pts2)   # covariance matrix
    U,S,V = np.linalg.svd(H)
    V = V.T
    R = V.dot(U.T)
    t = -R.dot(mean1.T) + mean2.T
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = t
    T[-1, -1] = 1
    return T

if __name__ == '__main__':
    # calculate_transformation()
    # filename = '/home/hwangmh/pycharmprojects/FLSpegtransfer/vision/coordinate_pairs.npy'
    # data = np.load(filename)
    # print(data)
    pts1 = [[0, 1, 0], [1, 0, 0], [0, -1, 0]]
    pts2 = [[-0.7071, 0.7071, 0], [0.7071, 0.7071, 0], [0.7071, -0.7071, 0]]
    T = get_rigid_transform(pts1, pts2)
    print(T)