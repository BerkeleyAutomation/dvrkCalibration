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

## Calibration Procedure

## Setup Fiducials
It goes 12 mm red ball at top, then slide 31 mm shaft, then 10 mm red ball (these balls will be slightly bigger than the PSM so wrap small clear piece of tape around so the balls don’t fall off such that the base of the red ball covers the intersection between the black and silver parts of the PSM). Then you grip the cross with the PSMs.
Insert picture of calibration here

### Random Sample Generation
This script will ask you which PSM you are calibrating. Then, ask if you need to find the bounding box values.
Usually, you will want to do this by putting in the plus sign fiducial and then freedriving the robot to the corners.
From there, you can call the get_pose function to get the robot pose at the corners. Then, update the parameters accordingly in the file.
It should output a npy file in calibration_outputs folder called prime_psmX_random_sampled.npy
```
cd ~/dvrkCalibration
source activate_calibration.bash
cd experiment/0_trajectory_extraction
python random_sample_generation.py
```

### Shallow Calibration (Camera to PSM/World Calibration)
Once you have your random trajectory, then run this. Make sure the cross fiducial IS NOT GRIPPED by the PSM. It should just be the two red balls on the PSM shaft. While the calibration is running, you will be able to see the end effector points as detected by the camera. Essentially, it finds the two balls and then applies the corresponding offset to get the gripper tip. If this offset seems incorrect, you can modify it in the dvrkArm.py file. This will output a psm_robot_to_zivid.npy which is the camera to robot transform along with the various points in camera space (psm1_pos_act.npy) and psm/robot/world space (psm1_pos_des.npy). It will also output timestamps, but this isn't too relevant (psm1_t_stamp_raw.npy)
```
cd ~/dvrkCalibration
source activate_calibration.bash
python dvrkShallowCalibration.py
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
