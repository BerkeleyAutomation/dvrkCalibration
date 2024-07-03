import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
from dotmap import DotMap
import threading
import cv2
Gst.init(None)

class gst_config():
    def __init__(self, config, pipeline):
        # Create the elements
        self.source = Gst.ElementFactory.make(config.source.name, None)
        self.source.set_property("mode", config.source.mode)
        self.source.set_property("connection", config.source.connection)
        self.source.set_property("device-number", config.source.device_number)

        self.convert = Gst.ElementFactory.make("videoconvert", None)
        self.caps = Gst.caps_from_string(config.caps)
        self.sink = Gst.ElementFactory.make("appsink", None)
        self.sink.set_property("emit-signals", True)
        self.sink.set_property("caps", self.caps)
        self.sink.connect("new-sample", self.buffer, self.sink)

        if not self.source or not self.sink or not pipeline:
            print("Not all elements could be created.")
            exit(-1)

        pipeline.add(self.source)
        pipeline.add(self.convert)
        pipeline.add(self.sink)
        self.img_arr = []

    def buffer(self, sink, data):
        sample = sink.emit("pull-sample")
        # buf = sample.get_buffer()
        # print "Timestamp: ", buf.pts
        self.img_arr = self.gst_to_opencv(sample)
        return Gst.FlowReturn.OK

    @classmethod
    def gst_to_opencv(cls, sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
        # print(caps.get_structure(0).get_value('format'))
        # print(caps.get_structure(0).get_value('height'))
        # print(caps.get_structure(0).get_value('width'))
        # print(buf.get_size())

        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
             caps.get_structure(0).get_value('width'),
             3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)
        return arr

    @classmethod
    def create_config(cls, which):
        config = DotMap()
        config.source.name = "decklinkvideosrc"
        config.source.mode = 0  # auto detect
        config.source.connection = 0  # SDI
        config.caps = "video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}"
        if which == "left":
            config.source.device_number = 1  # for left image
        elif which == "right":
            config.source.device_number = 0  # for right image
        return config


class Endoscope:
    def __init__(self):
        # Create the empty pipeline
        self.pipeline = Gst.Pipeline.new("test-pipeline")

        # Left gst object
        config = gst_config.create_config("left")
        self.gst_left = gst_config(config, self.pipeline)

        # Right gst object
        config = gst_config.create_config("right")
        self.gst_right = gst_config(config, self.pipeline)

        # Link
        if not Gst.Element.link(self.gst_left.source, self.gst_left.convert)\
                or not Gst.Element.link(self.gst_left.convert, self.gst_left.sink):
            print("Elements could not be linked.")
            exit(-1)

        if not Gst.Element.link(self.gst_right.source, self.gst_right.convert)\
                or not Gst.Element.link(self.gst_right.convert, self.gst_right.sink):
            print("Elements could not be linked.")
            exit(-1)

        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")
            exit(-1)

        # Wait until error or EOS
        self.bus = self.pipeline.get_bus()

        # Data members
        self._img_left = []
        self._img_right = []

    @property
    def img_left(self):
        self._img_left = self.gst_left.img_arr
        return self._img_left

    @property
    def img_right(self):
        self._img_right = self.gst_right.img_arr
        return self._img_right

    def exit(self):
        # Free resources
        self.pipeline.set_state(Gst.State.NULL)

    def parse_message(self):
        raise NotImplementedError
        # while True:
        #     message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
        #     # print "image_arr: ", image_arr
        #     if gst_left.img_arr is not None:
        #         cv2.imshow("Left image", self.gst_left.img_arr)
        #         cv2.waitKey(1)
        #     if gst_right.img_arr is not None:
        #         cv2.imshow("Right image", self.gst_right.img_arr)
        #         cv2.waitKey(1)
        #     if message:
        #         if message.type == Gst.MessageType.ERROR:
        #             err, debug = message.parse_error()
        #             print(("Error received from element %s: %s" % (
        #                 message.src.get_name(), err)))
        #             print(("Debugging information: %s" % debug))
        #             break
        #         elif message.type == Gst.MessageType.EOS:
        #             print("End-Of-Stream reached.")
        #             break
        #         elif message.type == Gst.MessageType.STATE_CHANGED:
        #             if isinstance(message.src, Gst.Pipeline):
        #                 old_state, new_state, pending_state = message.parse_state_changed()
        #                 print(("Pipeline state changed from %s to %s." %
        #                        (old_state.value_nick, new_state.value_nick)))
        #         else:
        #             print("Unexpected message received.")