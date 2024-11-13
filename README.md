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

## Repo Structure
This calibration was setup to work with the autonomous suturing project (STITCH) and our updated dvrk controller/vision library. It should be structured like this
```bash
.
├── automated_suturing
├── dvrk_2024
├── dvrkCalibration
```
## Calibration Procedure

## Setup Fiducials
It goes 12 mm red ball at top, then slide 31 mm shaft, then 10 mm red ball (these balls will be slightly bigger than the PSM so wrap small clear piece of tape around so the balls don’t fall off such that the base of the red ball covers the intersection between the black and silver parts of the PSM). Then you grip the cross with the PSMs.
Insert picture of calibration here

### Random Sample Generation
This script will ask you which PSM you are calibrating. Then, ask if you need to find the bounding box values.
Usually, you will want to do this by putting in the plus sign fiducial and then freedriving the robot to the corners.
From there, you can call the get_pose function to get the robot pose at the corners. Then, update the parameters accordingly in the file.
It should output a npy file in calibration_outputs folder called prime_psmX_random_sampled.npy. You should have different samples for shallow calibration and deep calibration because the cross fiducial marker prevents you from exploring ur space adequately. Also, to sanity check that your shallow calibration is going to work, do it for just 30 spots first before doing the full 2000, and make sure your error is <1 mm
```
cd ~/dvrkCalibration
source activate_calibration.bash
cd experiment/0_trajectory_extraction
python random_sample_generation.py
```

### Specific Sample Generation
While you are able to somewhat learn the hysteresis effect with random samples, we find that it is better to train on trajectories that best mimic the task that will be performed. In our case, we collected specific poses for each PSM for suturing and created samples off that.
```
cd ~/dvrkCalibration
source activate_calibration.bash
cd experiment/0_trajectory_extraction
python psm_suturing_sample_generation.py
```

### Shallow Calibration (Camera to PSM/World Calibration)
Once you have your random trajectory, then run this. Make sure the cross fiducial IS NOT GRIPPED by the PSM. It should just be the two red balls on the PSM shaft. While the calibration is running, you will be able to see the end effector points as detected by the camera. Essentially, it finds the two balls and then applies the corresponding offset to get the gripper tip. If this offset seems incorrect, you can modify it in the dvrkArm.py file. This will output a psm_robot_to_zivid.npy which is the camera to robot transform along with the various points in camera space (psm1_pos_act.npy) and psm/robot/world space (psm1_pos_des.npy). It will also output timestamps, but this isn't too relevant (psm1_t_stamp_raw.npy)
```
cd ~/dvrkCalibration
source activate_calibration.bash
python dvrkShallowCalibration.py
```

### Shallow Calibration Verification
To verify that the shallow calibration actually worked, there are 2 things you can do. First, you can quantify the error with the verification script as explained here. You should be able to get <1mm with this calibration.
```
cd ~/dvrkCalibration
source activate_calibration.bash
python cam_to_robot_verification.py
```

The other thing you can do is to check using the dvrk_2024 repo with the following. When you run this, you should see a green dot on the gripper tip (check the Zivid, the allied vision one won't work until you do the calibration specified later). As you free drive the PSM around, the gripper should follow the tip. This will work for moving q1,q2,q3 as they are not significantly affected by the cabling effect.
```
cd ~/automated_suturing
source activate_suturing.bash
cd ~/dvrk_2024/dvrk/vision
python calibration/calibration_robot_to_camera.py
```

### Allied Vision Camera Calibration
This is technically in an adjacent repo, but I wanted to put all the calibration instructions in the ssame spot. For the Allied Vision stereo camera, we capture checkerboard images, calculate intrinsics, and stereo calibrate all in one file as explained here. To verify the quality of the stereo calibration, the rectified checkerboard image pairs will show up epipolar lines and they should mainly be horizontal (there will be some funky images, but most of them should be horizontal)
```
cd ~/automated_suturing
source activate_suturing.bash
cd ~/dvrk_2024/dvrk/vision
python calibration/prime_allied_vision_checkerboard_calibration.py
```

### Zivid to Allied Vision Camera Calibration
Because we have a higher FOV and more accurate depth readings from the Zivid compared to the AlliedVision stereo cameras. To calibrate the stereo pair to the Zivid, we use an Aruco marker and place it in random spots in the workspace.
```
cd ~/automated_suturing
source activate_suturing.bash
cd ~/dvrk_2024/dvrk/vision
python calibration/prime_av_stereo_to_zivid_calibration.py
```

### Deep Calibration
After the Random Sample Generation, place the fiducial marker in the gripper of the PSM you want to calibrate, and it will go through the data collection process and save everything accordingly so it can then subsequently train.
```
cd ~/dvrkCalibration
source activate_calibration.bash
python dvrkCalibration.py
```

## Model Training
Training script to learn cabling effect. Note that you can choose between specific trajectory, random trajectory, or both.
```
conda activate dvrk_calibration_env
cd ~/dvrkCalibration/experiment/3_training/modeling
python train.py
```

## Evaluation
To see if the model works, you can have it run through some evaluation poses. Note, that currently they are hardocded to resemble poses that each PSM would go through while suturing, but you can change these to be whatever task is relevant to you.
```
cd ~/dvrkCalibration
source activate_calibration.bash
python dvrkCalibrationEvaluation.py
```

## Final Outputs
Here are where calibration outputs will be saved.
The calibration models and stereo camera calibration will be saved in the dvrk_2024 folder. The models saved here are automatically setup to work with the dvrk motion controller library and the calibration matrices are automatically setup to rectify the stereo images accordingly.

The rest of the files are saved in the dvrkCalibration folder.
```bash
.
├── automated_suturing
├── dvrk_2024
    ├── data
        ├── allied_vision_calibration_matrices (Stereo Checkerboard Calibration images and matrices)
    ├── dvrk
        ├── calibration_models
            ├── calibration_models_psm1
            ├── calibration_models_psm2
├── dvrkCalibration
    ├── experiment
        ├── 0_trajectory_extraction
            ├── allied_vision_calibration_outputs (Stereo Checkerboard Calibration images and matrices)
            ├── allied_vision_to_zivid_calibration_outputs (Aruco marker calibration for 2 diff cameras images and matrices)
            ├── model_outputs (Same as the model outputs saved in calibration models, but this will include intermediate values)
            ├── shallow_and_deep_calibration_outputs
                ├── psm1_robot_to_zivid.npy
                ├── psm2_robot_to_zivid.npy
```
