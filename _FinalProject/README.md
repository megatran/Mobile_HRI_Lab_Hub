## Robotic Medical Crash Cart

#### A CS6755 Final Project

Team: Nhan Tran, Frank Kim, Wendy Huang, Daniel Zhou

<p>
    <img src="./images/hospitalrobot.gif" width="50%" />
</p>



### Final Report
##### Please see `CS6755_Final_Project_Writeup.pdf` in this directory
- https://github.com/megatran/Mobile_HRI_Lab_Hub/blob/main/_FinalProject/CS6755_Final_Project_Writeup.pdf

#### Code
- Please see `mobilehri_ws/src` in this directory


### To start

We have two launch files for the robot prototype 

1. `ros2 launch mobile_robot_control hospital_robot_launch.py`

    - This set ups multiple nodes such as joystick, teleop controller, and speaker/sound interaction
    - Let’s launch this first to verify that we can use the joystick to control the robot (NOTE: always press joystick L1 to enable teleop drive control)

2. `ros2 launch yolov5_ros yolov5s_simple.launch.py`
    - This sets up a vision module such as the realsense camera node, and YOLO object/human detection node.
	- Note that our vision mode is VERY PRELIMINARY since the course project focuses more on the mobile aspect of HRI than automation.


To enable “vision mode”, press R1 on the joystick. When you walk towards the robot (or when a human is detected), you’ll see the wheels moving when you press the R1 button. When you walk out of the camera’s field of view, the wheels should stop moving (when you hold R1)


 <embed src="./CS6755_Final_Project_Writeup.pdf" width="600px" height="500px" />



