from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    """
    Reference topics for realsense

    /camera/accel/imu_info
    `/camera/accel/metadata
    /camera/accel/sample
    /camera/aligned_depth_to_color/camera_info
    /camera/aligned_depth_to_color/image_raw
    /camera/aligned_depth_to_infra1/camera_info
    /camera/aligned_depth_to_infra1/image_raw
    /camera/color/camera_info
    /camera/color/image_raw
    /camera/color/metadata
    /camera/depth/camera_info
    /camera/depth/image_rect_raw
    /camera/depth/metadata
    /camera/extrinsics/depth_to_accel
    /camera/extrinsics/depth_to_color
    /camera/extrinsics/depth_to_gyro
    /camera/extrinsics/depth_to_infra1
    /camera/extrinsics/depth_to_infra2
    /camera/gyro/imu_info
    /camera/gyro/metadata
    /camera/gyro/sample
    /camera/imu
    /camera/infra1/camera_info
    /camera/infra1/image_rect_raw
    /camera/infra1/metadata
    /camera/infra2/camera_info
    /camera/infra2/image_rect_raw
    /camera/infra2/metadata
    /cmd_vel
    /joy
    /joy/set_feedback
    /parameter_events
    /rosout
    /tf_static`
    """

    # FOR VISION PROCESSING, PLEASE RUN `ros2 launch yolov5_ros yolov5s_simple.launch.py `


    # realsense_camera_node = Node(
    #     package="realsense2_camera",
    #     executable="realsense2_camera_node",
    #     parameters=[
    #         {"align_depth.enable": True},
    #         {"rgb_camera.profile": '640x480x30'},
    #         {"depth_module.profile": '640x480x30'}
    #         ],
    #     namespace="camera"
    # )

    mobile_robot_control_node = Node(
            package='mobile_robot_control',
            executable='mobile_robot_control_node',
            name='robot_controll'
        )
    joy_teleop_keymapping_node = Node(
            package='joy_teleop_keymapping',
            executable='keymapping_node',
            name='keymap'
        )
    joy_node = Node(
            package='joy',
            executable='joy_node',
            name='joy',
        )

    # yolov5_ros = Node(
    #     package="yolov5_ros", executable="yolov5_ros",
    #     parameters=[
    #         {"view_img":True},
    #     ],
    # )    

    speaker_node = Node(
        package="speaker_interaction", executable="speaker_interaction",
    )

    return LaunchDescription([
        #realsense_camera_node,
        # yolov5_ros,
        speaker_node,
        mobile_robot_control_node,
        joy_teleop_keymapping_node,
        joy_node,
    ])