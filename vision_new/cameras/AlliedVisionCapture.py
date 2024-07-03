import os
import copy
import cv2
import rospy
import threading
from multiprocessing import Process, Queue
import queue
from typing import Optional
from vimba import *
import numpy as np
import time
from dvrk.vision.cameras.AlliedVisionUtils import AlliedVisionUtils
from dvrk.utils.ImgUtils import ImgUtils
import dvrk.vision.vision_constants as cst

# Camera Info.
# /// Name          : GC1290C (Left)
# /// Model         : GC1290C (02-2186A)
# /// S/N           : 02-2186A-06108
# /// ID            : DEV_000F310199C1
# /// Interface ID  : enp6s0
#
# /// Name          : GC1290C (Right)
# /// Model         : GC1290C (02-2186A)
# /// S/N           : 02-2186A-17617
# /// ID            : DEV_000F31021FD1
# /// Interface ID  : eno1


# Thread Objects
class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue, settings_fpath: str):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()
        self.settings_fpath = settings_fpath

    def __call__(self, cam: Camera, frame: Frame):
        # This method is executed within VimbaC context. All incoming frames
        # are reused for later frame acquisition. If a frame shall be queued, the
        # frame must be copied and the copy must be sent, otherwise the acquired
        # frame will be overridden as soon as the frame is reused.
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                self.try_put_frame(self.frame_queue, cam, frame_cpy)
        cam.queue_frame(frame)

    def try_put_frame(self, q: queue.Queue, cam: Camera, frame: Optional[Frame]):
        try:
            q.put_nowait((cam.get_id(), frame))
        except queue.Full:
            pass

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        # self.cam.get_feature_by_name("GVSPAdjustPacketSize").run()
        # self.cam.get_feature_by_name("PixelFormat").set("BGR8Packed")
        self.cam.load_settings(self.settings_fpath, PersistType.All)
        time.sleep(2.0)
        # self.cam.ExposureAuto.set("Off")

    def set_exposure(self, exposure_time):
        self.cam.ExposureTimeAbs.set(exposure_time)
        # print(f"set exposure to {exposure_time}")

    def run(self):
        self.log.info("Thread 'FrameProducer({})' started.".format(self.cam.get_id()))
        try:
            with self.cam:
                self.setup_camera()
                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                finally:
                    self.cam.stop_streaming()
        except VimbaCameraError:
            pass
        finally:
            self.try_put_frame(self.frame_queue, self.cam, None)
        self.log.info("Thread 'FrameProducer({})' terminated.".format(self.cam.get_id()))


class FrameConsumer(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, use_ROS=False, visualize=False, av_util: AlliedVisionUtils = None):
        threading.Thread.__init__(self)
        self.use_ROS = use_ROS
        self.visualize = visualize
        self.av_util = av_util
        if use_ROS:
            import rospy
            from sensor_msgs.msg import Image, CompressedImage, JointState
            from cv_bridge import CvBridge

            # create ROS node
            if not rospy.get_node_uri():
                rospy.init_node("StereoCamNode", anonymous=True, log_level=rospy.WARN)
            else:
                rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

            self.__image_left_pub = rospy.Publisher("/av/img_left", Image, latch=True, queue_size=1)
            self.__image_right_pub = rospy.Publisher("/av/img_right", Image, latch=True, queue_size=1)
            self.__bridge = CvBridge()
            self.__msg_left = Image()
            self.__msg_right = Image()

            if self.av_util is not None:
                self.__image_left_rect_pub = rospy.Publisher("/av/img_left_rect", Image, latch=True, queue_size=1)
                self.__image_right_rect_pub = rospy.Publisher("/av/img_right_rect", Image, latch=True, queue_size=1)
                self.__msg_left_rect = Image()
                self.__msg_right_rect = Image()

        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.cam_id_left = "DEV_000F310199C1"
        self.cam_id_right = "DEV_000F31021FD1"
        self.img_left = []
        self.img_right = []
        self.img_flag = False  # used once at starting
        self.killswitch = threading.Event()
        self.alive = True

    def run(self):
        # KEY_CODE_ENTER = 13
        frames = {}
        self.alive = True
        self.log.info("Thread 'FrameConsumer' started.")
        while self.alive:
            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

                # Add/Remove frame from current state.
                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Construct image by stitching frames together.
            if frames:
                for cam_id in sorted(frames.keys()):
                    # cv_image = frames[cam_id].as_opencv_image()
                    np_image = frames[cam_id].as_numpy_ndarray()
                    if cam_id == self.cam_id_left:
                        if self.use_ROS:
                            self.__msg_left = self.__bridge.cv2_to_imgmsg(np_image, "bgr8")
                            self.__msg_left.header.stamp = rospy.Time.now()
                            # self.__msg_left = self.__bridge.cv2_to_compressed_imgmsg(np_image, "jpeg, bgr8")
                            # self.msg_right = self.bridge.cv2_to_compressed_imgmsg(img_right, "jpeg, bgr8")
                            self.__image_left_pub.publish(self.__msg_left)
                            if self.av_util is not None:
                                np_image_rect = self.av_util.rectify_single(np_image, is_left=True)
                                self.__msg_left_rect = self.__bridge.cv2_to_imgmsg(np_image_rect, "bgr8")
                                self.__msg_left_rect.header.stamp = rospy.Time.now()
                                self.__image_left_rect_pub.publish(self.__msg_left_rect)
                        self.img_left = np_image
                    elif cam_id == self.cam_id_right:
                        if self.use_ROS:
                            self.__msg_right = self.__bridge.cv2_to_imgmsg(np_image, "bgr8")
                            self.__msg_right.header.stamp = rospy.Time.now()
                            # self.__msg_right = self.__bridge.cv2_to_compressed_imgmsg(np_image, "jpeg, bgr8")
                            self.__image_right_pub.publish(self.__msg_right)
                            if self.av_util is not None:
                                np_image_rect = self.av_util.rectify_single(np_image, is_left=False)
                                self.__msg_right_rect = self.__bridge.cv2_to_imgmsg(np_image_rect, "bgr8")
                                self.__msg_right_rect.header.stamp = rospy.Time.now()
                                self.__image_right_rect_pub.publish(self.__msg_right_rect)
                        self.img_right = np_image

            if len(self.img_left) == 0 or len(self.img_right) == 0:
                time.sleep(0.01)  # delay for quick start of streaming
            else:
                # cv2.imwrite("img_pegboard_right_raw.png", self.img_right)
                # cv2.imwrite("img_pegboard_left_raw.png", self.img_left)
                if self.visualize:
                    stacked = ImgUtils.stack_stereo_img(self.img_left, self.img_right, scale=0.7)
                    cv2.imshow("stereo images", stacked)
                    cv2.waitKey(1)
            time.sleep(0.01)

            # # Check for shutdown condition
            # if KEY_CODE_ENTER == cv2.waitKey(10):
            #     cv2.destroyAllWindows()
            #     alive = False
        self.log.info("Thread 'FrameConsumer' terminated.")

    def stop(self):
        self.alive = False


class AlliedVisionCapture(threading.Thread):
    def __init__(self, use_ROS=True, visualize=False):
        threading.Thread.__init__(self)
        self.av_util = AlliedVisionUtils()

        # threads for capturing
        self.frame_queue = queue.Queue(maxsize=10)
        self.producers = {}
        self.producers_lock = threading.Lock()
        self.consumer = {}
        self.consumer = FrameConsumer(self.frame_queue, use_ROS=use_ROS, visualize=visualize, av_util=self.av_util)
        # self.start()

    def __call__(self, cam: Camera, event: CameraEvent):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == CameraEvent.Detected:
            with self.producers_lock:
                settings_file = os.path.join(cst.AV_SETTINGS_PATH, f"{cam.get_id()}_settings.xml")
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue, settings_file)
                self.producers[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()

    def run(self):
        log = Log.get_instance()
        vimba = Vimba.get_instance()
        vimba.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)
        log.info("Thread 'MainThread' started.")
        with vimba:
            # Construct FrameProducer threads for all detected cameras
            for cam in vimba.get_all_cameras():
                settings_file = os.path.join(cst.AV_SETTINGS_PATH, f"{cam.get_id()}_settings.xml")
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue, settings_file)
                # with cam:
                # settings_file = "{}_settings.xml".format(cam.get_id())
                # cam.save_settings(settings_file, PersistType.All)
                # print(f"saved to {settings_file}")
                # settings_file = os.path.join(cst.AV_SETTINGS_PATH, f"{cam.get_id()}_settings.xml")
                # cam.load_settings(settings_file, PersistType.All)

            # Start FrameProducer threads
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            # Start and wait for consumer to terminate
            vimba.register_camera_change_handler(self)
            self.consumer.start()
            self.consumer.join()
            vimba.unregister_camera_change_handler(self)

            # Stop all FrameProducer threads
            with self.producers_lock:
                # Initiate concurrent shutdown
                for producer in self.producers.values():
                    producer.stop()

                # Wait for shutdown to complete
                for producer in self.producers.values():
                    producer.join()
        log.info("Thread 'MainThread' terminated.")

    def capture(self, which="rectified"):
        if which == "original":
            return self.consumer.img_left, self.consumer.img_right
        elif which == "rectified":
            if len(self.consumer.img_left) == 0 or len(self.consumer.img_right) == 0:
                return [], []
            else:
                return self.av_util.rectify(self.consumer.img_left, self.consumer.img_right)

    def stop(self):
        self.consumer.stop()


if __name__ == "__main__":
    av = AlliedVisionCapture(use_ROS=True, visualize=False)
    print("AV stereo ready")
    img_left, img_right = av.capture(which="rectified")

    print(img_left)
    # img_left = np.array(img_left)
    # print(img_left.shape)
    # print(img_left.dtype)

    # cv2.imshow("img_left", img_left)
    # cv2.imshow("img_right", img_right)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # np.save("av_left.np", img_left)
    # np.save("av_right.np", img_right)

    # np.save("/home/davinci/Desktop/av_left.np", img_left)
    # np.save("/home/davinci/Desktop/av_right.np", img_right)

    # cv2.imwrite("/home/davinci/Desktop/av_left.png", img_left)
    # cv2.imwrite("/home/davinci/Desktop/av_right.png", img_right)

    # while True:
    #     time.sleep(1.0)
    # print()
    # import numpy as np
    # from VisualServoing.utils.ImgUtils import ImgUtils

    # counter = 0

    # while True:
    #     time.sleep(1)
    #     img_left, img_right = av.capture(which="rectified")
    #     if len(img_left) == 0 or len(img_right) == 0:
    #         pass
    #         print("no img")
    #     else:
    #         # stereo = ImgUtils.compare_rectified_img(img_left, img_right, scale=0.7, line_gap=30)
    #         stereo = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
    #         cv2.imshow("stereo_image", stereo)
    #         cv2.waitKey(1)
    #         cv2.imwrite("../trash/stereo_sphere_cal_left.png", img_left)
    #         cv2.imwrite("../trash/stereo_sphere_cal_right.png", img_right)
    #     print("loop")
    #     time.sleep(0.01)
    #     counter += 1
    #     if counter > 3:
    #         av.stop()
