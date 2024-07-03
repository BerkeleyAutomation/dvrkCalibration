import os

# directories and paths
ROOT_PATH = "/home/davinci/dvrk"
DATA_DIR = "data"
DATA_PATH = os.path.join(ROOT_PATH, DATA_DIR)
VISION_PATH = os.path.join(ROOT_PATH, "dvrk/vision")
AV_SETTINGS_PATH = os.path.join(DATA_PATH, "av_settings")

IMG_PREFIX = "img_"
IMG_EXTENSION = ".png"

ZIVID = "zivid"
ALLIED_VISION = "allied_vision"
ALLIED_VISION_SINGLE = "allied_vision_single"
ZED = "zed"

# calibration matrices
K_MAT_FNAME = "K_mat.npy"
D_MAT_FNAME = "D_mat.npy"
R_STEREO_MAT_FNAME = "R_STEREO_mat.npy"
T_STEREO_MAT_FNAME = "T_STEREO_mat.npy"
E_STEREO_MAT_FNAME = "E_STEREO_mat.npy"
F_STEREO_MAT_FNAME = "F_STEREO_mat.npy"
R_MAT_FNAME = "R_mat.npy"
P_MAT_FNAME = "P_mat.npy"
Q_STEREO_MAT_FNAME = "Q_STEREO_mat.npy"
MAPX_MAT_FNAME = "MAPX_mat.npy"
MAPY_MAT_FNAME = "MAPY_mat.npy"

# Allied Vision
AV_IMG_SHAPE = (1280, 960)
