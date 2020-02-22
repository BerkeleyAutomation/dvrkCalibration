import socket
import struct
import numpy as np
import threading
import FLSpegtransfer.utils.CmnUtil as U

class dvrkMotionBridgeP():
    def __init__(self):
        # Data members
        self.act_pos1 = []
        self.act_rot1 = []
        self.act_jaw1 = []
        self.act_pos2 = []
        self.act_rot2 = []
        self.act_jaw2 = []

        # UDP setting
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT_SERV = 1215
        self.UDP_PORT_SERV2 = 1216  # auxiliary channel for sending actual value
        self.UDP_PORT_CLNT = 1217
        self.UDP_PORT_CLNT2 = 1218
        self.buffer_size = 1024
        self.sock = socket.socket(socket.AF_INET,  # Internet
                             socket.SOCK_DGRAM)  # UDP
        self.sock2 = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP
        self.sock.setblocking(1)  # Blocking mode
        self.sock.bind((self.UDP_IP, self.UDP_PORT_CLNT))
        self.sock2.bind((self.UDP_IP, self.UDP_PORT_CLNT2))

        # Thread
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while True:
            data_recv, addr = self.sock2.recvfrom(self.buffer_size)  # buffer size is 1024 bytes
            data_unpack = list(struct.unpack('ffffffffffffffff', data_recv))
            self.act_pos1 = data_unpack[0:3]
            self.act_rot1 = data_unpack[3:7]
            self.act_jaw1 = data_unpack[7:8]
            self.act_pos2 = data_unpack[8:11]
            self.act_rot2 = data_unpack[11:15]
            self.act_jaw2 = data_unpack[15:16]

    def set_pose(self, pos1=[], rot1=[], jaw1=[], pos2=[], rot2=[], jaw2=[]):
        # data sending
        input_flag = [True, True, True, True, True, True]
        if pos1==[]:
            pos1 = [0.0, 0.0, -0.13]
            input_flag[0] = False
        if rot1==[]:
            rot1 = [0.0, 0.0, 0.0, 1.0]
            input_flag[1] = False
        if jaw1==[]:
            jaw1 = [0.0]
            input_flag[2] = False
        if pos2==[]:
            pos2 = [0.0, 0.0, -0.13]
            input_flag[3] = False
        if rot2==[]:
            rot2 = [0.0, 0.0, 0.0, 1.0]
            input_flag[4] = False
        if jaw2==[]:
            jaw2 = [0.0]
            input_flag[5] = False
        data_send = struct.pack('ffffffffffffffff??????', pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2], rot1[3], jaw1[0],
                                                          pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2], rot2[3], jaw2[0],
                                                          input_flag[0], input_flag[1], input_flag[2], input_flag[3], input_flag[4], input_flag[5])
        self.sock.sendto(data_send, (self.UDP_IP, self.UDP_PORT_SERV))

        # data receiving
        data, _ = self.sock.recvfrom(self.buffer_size)
        goal_reached = list(struct.unpack('?', data))
        return goal_reached[0]

    def set_position(self, pos1=[], pos2=[]):
        # data sending
        input_flag = [True, True]
        if pos1 == []:
            pos1 = [0.0, 0.0, -0.13]
            input_flag[0] = False
        if pos2 == []:
            pos2 = [0.0, 0.0, -0.13]
            input_flag[1] = False
        data_send = struct.pack('ffffffffffffffff??????', pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2],
                                input_flag[0], input_flag[1])
        self.sock.sendto(data_send, (self.UDP_IP, self.UDP_PORT_SERV))

        # data receiving
        data, _ = self.sock.recvfrom(self.buffer_size)
        goal_reached = list(struct.unpack('?', data))
        return goal_reached[0]

if __name__ == "__main__":
    perception = dvrkMotionBridgeP()
    pos1 = [0.0, 0.0, -0.13]
    rot1 = [0.0, 0.0, 0.0]
    q1 = U.euler_to_quaternion(rot1, unit='deg')
    jaw1 = [0.0]
    pos2 = [0.0, 0.0, -0.13]
    rot2 = [0.0, 0.0, 0.0]
    q2 = U.euler_to_quaternion(rot2, unit='deg')
    jaw2 = [0.0]
    t = 0.0
    while True:
        t += 0.01
        pos1[0] = 0.1*np.sin(2*3.14*t)
        pos2[0] = 0.1*np.sin(2*3.14*t)
        jaw1[0] = 0.6*np.sin(2*3.14*2*t) + 0.6
        jaw2[0] = 0.6*np.sin(2*3.14*2*t) + 0.6
        perception.set_pose(pos1=pos1, rot1=q1, jaw1=jaw1, pos2=pos2, rot2=q2, jaw2=jaw2)