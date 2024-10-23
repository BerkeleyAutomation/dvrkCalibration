# Efficiently Calibrating Cable-Driven Robots and Autonomous Peg Transfer Task

## Installation
```
conda create --name dvrk_calibration_env python=3.8.10
pip install numpy==1.24.4
pip install opencv-python==4.6.0.66
pip install matplotlib==3.7.5
pip install pyyaml==6.0.1
pip install rospkg==1.5.1
pip install scipy==1.10.1
pip install open3d==0.18.0
pip install dotmap==1.3.30
```

## Training
```
conda activate dvrk_calibration_env
cd ~/dvrkCalibration/experiment/3_training/modeling
python train.py
```

## Inference
```
conda activate dvrk_calibration_env
cd /home/davinci/dvrkCalibration/experiment/4_verification
python test_inference.py
```
