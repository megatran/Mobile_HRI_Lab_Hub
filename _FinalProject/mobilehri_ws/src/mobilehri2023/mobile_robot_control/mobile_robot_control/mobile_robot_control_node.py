import rclpy
from mobile_robot_control.odrive_command import odrive_command

def main(args=None):
    rclpy.init(args=args)
    oc = odrive_command()
    try:
        rclpy.spin(oc)
    finally:
        oc.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
