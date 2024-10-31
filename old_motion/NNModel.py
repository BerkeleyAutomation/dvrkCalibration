from keras.models import model_from_json
import copy
import numpy as np

class NNModel():
    def __init__(self, ensemble, horizon, iter, alpha, model='forward'):
        # load model
        root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
        if model=='forward':
            model_dir = root + "experiment/4_verification/model/RNN_forward_peg_traj/"
            self.use_forward_model = True
        elif model=='inverse':
            model_dir = root + "experiment/4_verification/model/RNN_inverse_peg_traj/"
            self.use_forward_model = False
        else:
            print ("wrong argument input")
            exit()
        self.models = self.load_models(model_dir=model_dir, num_ensemble=ensemble)  # num_ensemble = 0, 3, 10

        # data members
        self.movement_cache = []
        self.horizon = horizon
        self.iter = iter
        self.alpha = alpha

    def load_models(self, model_dir, num_ensemble):
        models = []
        for i in range(1, num_ensemble + 1):
            json_file = open(model_dir + str(i) + '/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()

            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_dir + str(i) + "/model.h5")
            models.append(loaded_model)
        print("Loaded " + str(len(models)) + " models from disk.")
        return models

    def step(self, desired):
        self.add_to_cache(desired)
        q_cmd = self.get_next_cmd()
        self.movement_cache[-1] = q_cmd
        return q_cmd

    def add_to_cache(self, desired):    # [q(t-H+1), ..., q(t)]
        if len(self.movement_cache) < self.horizon:
            self.movement_cache.append(desired)
        else:
            self.movement_cache.pop(0)
            self.movement_cache.append(desired)

    def get_next_cmd(self):
        if len(self.movement_cache) < self.horizon:
            # print('not using history')
            return self.get_desired_pos()
        else:
            if self.use_forward_model:
                # print('using history')
                return self.get_delta_cmd(self.alpha, self.iter)
            else:
                q123 = self.get_desired_pos()[:3]
                q456 = self.get_inverse_cmd()
                joints = np.concatenate((q123, q456))
                return joints

    def get_delta_cmd(self, alpha, iter):
        assert self.use_forward_model
        q_targ = self.get_desired_pos()
        q_cmd = self.get_desired_pos()
        for i in range(iter):
            q_est = q_cmd + np.concatenate(([0., 0., 0.], self.invoke_model(q_cmd)))
            e_h = q_targ - q_est
            q_cmd = q_cmd + alpha * e_h
        return q_cmd

    def get_inverse_cmd(self):
        assert not self.use_forward_model
        q_targ = self.get_desired_pos()
        return self.invoke_model(q_targ)

    def get_desired_pos(self):
        return copy.deepcopy(self.movement_cache[-1])

    def invoke_model(self, q_cmd):
        inp = copy.deepcopy(self.movement_cache)
        inp[-1] = q_cmd
        inp = [d[3:] for d in inp]
        inp = np.expand_dims(inp, axis=0)
        outputs = []
        for model in self.models:
            outputs.append(model.predict(inp)[0])
        # print("STD:", np.std(outputs))
        # print("MED:", np.median(outputs, axis=0))
        # print("AVG:", np.mean(outputs, axis=0))
        output = np.median(outputs, axis=0)
        return output

if __name__ == "__main__":
    # from google.colab import drive
    # from sklearn.metrics import mean_squared_error
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import axes3d
    # import FLSpegtransfer.utils.CmnUtil as U
    NN = NNModel(ensemble=3, horizon=5, iter=3, alpha=0.5, model='forward')
