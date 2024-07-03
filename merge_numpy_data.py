import numpy as np
q_act_raw_data_300 = np.load('/home/davinci/dvrkCalibration/q_act_raw_300.npy')# 299 data points
q_act_raw_data_400 = np.load('/home/davinci/dvrkCalibration/q_act_raw_400.npy')# 99 data points
q_act_raw_data_450 = np.load('/home/davinci/dvrkCalibration/q_act_raw_450.npy')# 49 data points
q_act_raw_data_700 = np.load('/home/davinci/dvrkCalibration/q_act_raw_700.npy')# 199 data points
q_act_raw_data_950 = np.load('/home/davinci/dvrkCalibration/q_act_raw_950.npy') # 199 data points
q_act = np.vstack((q_act_raw_data_300,q_act_raw_data_400,q_act_raw_data_450,q_act_raw_data_700,q_act_raw_data_950))
np.save('/home/davinci/dvrkCalibration/data/psm2_q_act_raw_845.npy',q_act)

q_des_raw_data_300 = np.load('/home/davinci/dvrkCalibration/q_des_raw_300.npy')# 299 data points
q_des_raw_data_400 = np.load('/home/davinci/dvrkCalibration/q_des_raw_400.npy')# 99 data points
q_des_raw_data_450 = np.load('/home/davinci/dvrkCalibration/q_des_raw_450.npy')# 49 data points
q_des_raw_data_700 = np.load('/home/davinci/dvrkCalibration/q_des_raw_700.npy')# 199 data points
q_des_raw_data_950 = np.load('/home/davinci/dvrkCalibration/q_des_raw_950.npy') # 199 data points
q_des = np.vstack((q_des_raw_data_300,q_des_raw_data_400,q_des_raw_data_450,q_des_raw_data_700,q_des_raw_data_950))
np.save('/home/davinci/dvrkCalibration/data/psm2_q_des_raw_845.npy',q_des)