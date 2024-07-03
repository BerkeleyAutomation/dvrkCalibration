import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics

BD = BallDetectionRGBD()
zivid = ZividCapture()
# zivid.start()

while True:
    color = np.load('ball_img_zivid_color.npy')
    point = np.load('ball_img_zivid_point.npy')
    # point = np.load('fiducial_images/0/point.npy')
    # color, depth, point = zivid.capture_3Dimage(color='BGR')

    # Find balls
    pbr = BD.find_balls(color, point, 'red')
    pbg = BD.find_balls(color, point, 'green')
    pbb = BD.find_balls(color, point, 'blue')
    pby = BD.find_balls(color, point, 'yellow')

    pbr = np.array(pbr)
    d1 = np.linalg.norm(pbr[0][:3] - pbr[1][:3])
    d2 = np.linalg.norm(pbr[0][:3] - pbr[2][:3])
    d3 = np.linalg.norm(pbr[1][:3] - pbr[2][:3])
    print(d1, d2, d3)

    # visualize
    color = BD.overlay_ball(color, pbr)
    color = BD.overlay_ball(color, [pbg])
    color = BD.overlay_ball(color, [pbb])
    color = BD.overlay_ball(color, [pby])

    cv2.imshow("", color)
    cv2.waitKey(0)


    # Find tool position, joint angles, and overlay
    if pbr[0]==[] or pbr[1]==[]:
        pass
    else:
        pt = BD.find_tool_position(pbr[0], pbr[1])    # tool position of pitch axis
        pt = np.array(pt) * 0.001  # (m)
        pt = BD.Rrc.dot(pt) + BD.trc
        q1, q2, q3 = dvrkKinematics.ik_position(pt)
        # print(q0*180/np.pi, q2*180/np.pi, q3)
        color = BD.overlay_tool_position(color, [q1,q2,q3], (0,255,0))

        # Find tool orientation, joint angles, and overlay
        if len(pbr) < 3:
            pass
        elif [pbr[2], pbg, pbb, pby].count([]) < 3:
            pass
        else:
            Rm = BD.find_tool_orientation(pbr[2], pbg, pbb, pby)    # orientation of the marker
            q4,q5,q6 = dvrkKinematics.ik_orientation(q1,q2,Rm)
            # print(q4*180/np.pi,q5*180/np.pi,q6*180/np.pi)
            # print(q5*180/np.pi)
            color = BD.overlay_tool(color, [q1, q2, q3, q4, q5, q6], (0,255,0))

    cv2.imshow("images", color)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break