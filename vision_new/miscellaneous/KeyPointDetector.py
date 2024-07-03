import numpy as np
import os.path
from pathlib import Path
import tensorflow as tf
vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf
# from deeplabcut.pose_estimation_tensorflow.nnet import predict, predict_multianimal
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net


class KeyPointDetector:
    def __init__(self, config, shuffle=1, trainingsetindex=0, batchsize=None, TFGPUinference=True, modelprefix="",):
        # init config.
        self.cfg, self.dlc_cfg = self.config_model(config, shuffle, trainingsetindex, batchsize, modelprefix)
        self.TFGPUinference = TFGPUinference
        self.config_session()

    def config_model(self, config, shuffle=1, trainingsetindex=0, batchsize=None, modelprefix="",):
        tf.reset_default_graph()
        cfg = auxiliaryfunctions.read_config(config)
        trainFraction = cfg["TrainingFraction"][trainingsetindex]
        modelfolder = os.path.join(
            cfg["project_path"],
            str(auxiliaryfunctions.GetModelFolder(trainFraction, shuffle, cfg, modelprefix=modelprefix)),
        )
        path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
        try:
            dlc_cfg = load_config(str(path_test_config))
        except FileNotFoundError:
            raise FileNotFoundError(
                "It seems the model for shuffle %s and trainFraction %s does not exist."
                % (shuffle, trainFraction)
            )

        # Check which snapshots are available and sort them by # iterations
        try:
            Snapshots = np.array(
                [
                    fn.split(".")[0]
                    for fn in os.listdir(os.path.join(modelfolder, "train"))
                    if "index" in fn
                ]
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
                % (shuffle, shuffle)
            )

        if cfg["snapshotindex"] == "all":
            print(
                "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
            )
            snapshotindex = -1
        else:
            snapshotindex = cfg["snapshotindex"]

        increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

        ##################################################
        # Load and setup CNN part detector
        ##################################################
        # Check if data already was generated:
        dlc_cfg["init_weights"] = os.path.join(
            modelfolder, "train", Snapshots[snapshotindex]
        )
        # Update number of output and batchsize
        dlc_cfg["num_outputs"] = cfg.get("num_outputs", dlc_cfg.get("num_outputs", 1))

        if batchsize == None:
            # update batchsize (based on parameters in config.yaml)
            dlc_cfg["batch_size"] = cfg["batch_size"]
        else:
            dlc_cfg["batch_size"] = batchsize
            cfg["batch_size"] = batchsize

        if "multi-animal" in dlc_cfg["dataset_type"]:
            dynamic = (False, 0.5, 10)  # setting dynamic mode to false
            TFGPUinference = False
        return cfg, dlc_cfg

    def config_session(self):
        if self.TFGPUinference:
            tf.reset_default_graph()
            self.inputs = tf.placeholder(tf.float32, shape=[self.dlc_cfg.batch_size, None, None, 3])
            net_heads = pose_net(self.dlc_cfg).inference(self.inputs)
            self.outputs = [net_heads["pose"]]
            restorer = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # Restore variables from disk.
            restorer.restore(self.sess, self.dlc_cfg.init_weights)
        else:
            TF.reset_default_graph()
            self.inputs = TF.placeholder(tf.float32, shape=[self.dlc_cfg.batch_size, None, None, 3])
            net_heads = pose_net(self.dlc_cfg).test(self.inputs)
            self.outputs = [net_heads["part_prob"]]
            if self.dlc_cfg.location_refinement:
                self.outputs.append(net_heads["locref"])
            if ("multi-animal" in self.dlc_cfg.dataset_type) and self.dlc_cfg.partaffinityfield_predict:
                print("Activating extracting of PAFs")
                self.outputs.append(net_heads["pairwise_pred"])
            restorer = TF.train.Saver()
            self.sess = TF.Session()
            self.sess.run(TF.global_variables_initializer())
            self.sess.run(TF.local_variables_initializer())

            # Restore variables from disk.
            restorer.restore(self.sess, self.dlc_cfg.init_weights)

    def predict_key_points(self, image):
        pose_tensor = self.outputs[0]
        pose = self.sess.run(pose_tensor, feed_dict={self.inputs: np.expand_dims(image, axis=0).astype(float)}, )
        pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
        return pose