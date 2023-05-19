import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import subprocess

class InteractionSpeaker(Node):
    def __init__(self):
        super().__init__('Speaker_node')
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joyCallback, 10)
        # self.interaction_sub = self.create_subscription(String, '/interaction_sound', self.interactionCallback, 10)


    def joyCallback(self, msg):
        self.send_joy_Command(msg)
        
    # def interactionCallback(self, msg):
    #     self.send_interaction_Command(msg)

    def send_joy_Command(self, msg):
        t = Twist()
        # safety lock, press top left button
        if msg.buttons[1] == 1.0: # 
          subprocess.run(["espeak","I am following you."])
        elif msg.buttons[3] == 1.0:
          subprocess.run(["espeak","Ok, stop following. Have a nice day"])
        elif msg.buttons[2] == 1.0:
            subprocess.run(["espeak","Hi, I am a hospital robot. Please say follow me!"])
        elif msg.buttons[0]==1.0:
            subprocess.run(["espeak","I am too close to you. I will slow down"])
        
    # def send_interaction_Command(self, msg):
    #     t = Twist()
    #     # safety lock, press top left button
    #     if msg == "person show up":
    #         subprocess.run(["espeak","Hi, I am a hospital robot. Please say follow me or press R1, press L1 to terminate"])
    #     elif msg == "close":
    #         subprocess.run(["espeak","I am too close to you. I will slow down"])
    #     elif msg == "far":
    #         subprocess.run(["espeak","I am too far from you. I will speed up"])

def main(args = None):
    rclpy.init(args=args)
    ttj = InteractionSpeaker()
    rclpy.spin(ttj)
    rclpy.shutdown()


if __name__ == '__main__':
    main()