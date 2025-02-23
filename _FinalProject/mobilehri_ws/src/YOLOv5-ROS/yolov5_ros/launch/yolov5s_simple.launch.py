import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolov5_ros')

    camera=launch_ros.actions.Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        parameters=[{'align_depth.enable' :True}],
        namespace="camera"       

    )

    yolov5_ros = launch_ros.actions.Node(
        package="yolov5_ros", executable="yolov5_ros",
        parameters=[
            {"view_img":True},
        ],

    )


    return launch.LaunchDescription([
        camera,
        yolov5_ros
    ])
