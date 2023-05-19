import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import math
import torch
import torch.backends.cudnn as cudnn

from yolov5_ros.models.common import DetectMultiBackend
from yolov5_ros.utils.datasets import IMG_FORMATS, VID_FORMATS
from yolov5_ros.utils.general import (LOGGER, check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5_ros.utils.plots import Annotator, colors
from yolov5_ros.utils.torch_utils import select_device, time_sync
from yolov5_ros.utils.datasets import letterbox

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import Header
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import subprocess


import pyrealsense2.pyrealsense2 as rs

class yolov5_demo():
    def __init__(self,  weights,
                        data,
                        imagez_height,
                        imagez_width,
                        conf_thres,
                        iou_thres,
                        max_det,
                        device,
                        view_img,
                        classes,
                        agnostic_nms,
                        line_thickness,
                        half,
                        dnn,
                        target
                        ):
        self.weights = weights
        self.data = data
        self.imagez_height = imagez_height
        self.imagez_width = imagez_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn
        self.target=target
        # cmd_vel 
        
       
        self.s = str()
        
        self.load_model()

        

    def load_model(self):
        imgsz = (self.imagez_height, self.imagez_width)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        source = 0
        # Dataloader
        webcam = True
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True
        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    # callback ==========================================================================

    # return ---------------------------------------
    # 1. class (str)                                +
    # 2. confidence (float)                         +
    # 3. x_min, y_min, x_max, y_max (float)         +
    # ----------------------------------------------
    def image_callback(self, image_raw):
        class_list = []
        confidence_list = []
        x_min_list = []
        y_min_list = []
        x_max_list = []
        y_max_list = []
        person_xmin=[]
        person_xmax=[]
        person_ymin=[]
        person_ymax=[]
        persons=[]
        # im is  NDArray[_SCT@ascontiguousarray
        # im = im.transpose(2, 0, 1)
        self.stride = 32  # stride
        self.img_size = 640
        img = letterbox(image_raw, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        save_dir = "runs/detect/exp7"
        path = ['0']

        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):
            im0 = image_raw
            self.s += f'{i}: '

            # p = Path(str(p))  # to Path
            self.s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if self.target is not None:
                    annotator.box_label(self.target,'target',color=(0,0,255))
                else:
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        save_conf = False
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{self.names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        if 'person' in label:
                            # return cmd velocity by PID control
                            
                            person_xmin.append(xyxy[0].item())
                            person_ymin.append(xyxy[1].item())
                            person_xmax.append(xyxy[2].item())
                            person_ymax.append(xyxy[3].item())                        
                            
                        class_list.append(self.names[c])
                        confidence_list.append(conf)
                        # tensor to float
                        x_min_list.append(xyxy[0].item())
                        y_min_list.append(xyxy[1].item())
                        x_max_list.append(xyxy[2].item())
                        y_max_list.append(xyxy[3].item())
                
            for i in range(len(person_xmin)):     
                single =[person_xmin[i],person_ymin[i],person_xmax[i],person_ymax[i]]
                persons.append(single)       
            # Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.imshow("yolov5", im0)
                cv2.waitKey(1)  # 1 millisecond
            
            return class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list, persons

class yolov5_ros(Node):
    def __init__(self):
        super().__init__('yolov5_ros')

        self.bridge = CvBridge()

        self.pub_bbox = self.create_publisher(BoundingBoxes, 'yolov5/bounding_boxes', 10)
        # self.pub_image = self.create_publisher(Image, 'yolov5/image_raw', 10)
        self.twist_pub = self.create_publisher(Twist,'/cmd_vision',10)

        self.sub_image = self.create_subscription(Image, 'camera/color/image_raw', self.image_callback,10)
        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_callback,10)
        self.sub_aligned_depth = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.aligned_depth_image_callback,10)
        self.joy_command = None
        self.robot_stream_depth=None

        ## Pyrealsense instrics of the depth camera taken from data sheet
        self.depth_intrinsic = rs.intrinsics()
        # self.depth_intrinsic.width = 640
        # self.depth_intrinsic.height = 480
        # self.depth_intrinsic.ppx = 322.043121337891
        # self.depth_intrinsic.ppy = 238.831329345703
        # self.depth_intrinsic.fx = 393.181854248047
        # self.depth_intrinsic.fy = 393.181854248047

        """
        ros2 topic echo /camera/aligned_depth_to_color/camera_info
        header:
        stamp:
            sec: 1683394402
            nanosec: 774493952
        frame_id: camera_color_optical_frame
        height: 480
        width: 640
        distortion_model: plumb_bob
        d:
        - 0.0
        - 0.0
        - 0.0
        - 0.0
        - 0.0
        k:
        - 611.18505859375
        - 0.0
        - 317.17352294921875
        - 0.0
        - 610.2504272460938
        - 235.53453063964844
        - 0.0
        - 0.0
        - 1.0
        r:
        - 1.0
        - 0.0
        - 0.0
        - 0.0
        - 1.0
        - 0.0
        - 0.0
        - 0.0
        - 1.0
        p:
        - 611.18505859375
        - 0.0
        - 317.17352294921875
        - 0.0
        - 0.0
        - 610.2504272460938
        - 235.53453063964844
        - 0.0
        - 0.0
        - 0.0
        - 1.0
        - 0.0
        binning_x: 0
        binning_y: 0
        roi:
        x_offset: 0
        y_offset: 0
        height: 0
        width: 0
        do_rectify: false

        """
        self.depth_intrinsic.width = 640
        self.depth_intrinsic.height = 480
        self.depth_intrinsic.ppx = 317.17352294921875
        self.depth_intrinsic.ppy = 235.53453063964844
        self.depth_intrinsic.fx = 611.18505859375
        self.depth_intrinsic.fy = 610.2504272460938

        self.depth_intrinsic.model = rs.distortion.brown_conrady
        self.depth_intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Robot model (radius, wheelbase, etc)
        self.R = 0.07
        self.tyre_circumference = 2*math.pi*self.R
        # Adjust the wheel track to account for your robot
        self.wheel_track = 0.51

        self.target = None
        self.target_z = 100.0
        self.count = 0
        self.start_process=True
        self.xyz = None
        self.view=None
        # parameter
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

        self.declare_parameter('weights', str(ROOT) + '/config/yolov5s.pt')
        self.declare_parameter('data', str(ROOT) + '/data/coco128.yaml')
        self.declare_parameter('imagez_height', 480)
        self.declare_parameter('imagez_width', 640)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 100)
        self.declare_parameter('device', '0')
        self.declare_parameter('view_img', False)
        self.declare_parameter('classes', None)
        self.declare_parameter('agnostic_nms', False)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('half', False)
        self.declare_parameter('dnn', False)

        self.weights = self.get_parameter('weights').value
        self.data = self.get_parameter('data').value
        self.imagez_height = self.get_parameter('imagez_height').value
        self.imagez_width = self.get_parameter('imagez_width').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.view_img = self.get_parameter('view_img').value
        self.classes = self.get_parameter('classes').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.half = self.get_parameter('half').value
        self.dnn = self.get_parameter('dnn').value

        self.yolov5 = yolov5_demo(self.weights,
                                self.data,
                                self.imagez_height,
                                self.imagez_width,
                                self.conf_thres,
                                self.iou_thres,
                                self.max_det,
                                self.device,
                                self.view_img,
                                self.classes,
                                self.agnostic_nms,
                                self.line_thickness,
                                self.half,
                                self.dnn,
                                self.target)

         
    def yolovFive2bboxes_msgs(self, bboxes:list, scores:list, cls:list, img_header:Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        print(bboxes)
        # print(bbox[0][0])
        i = 0
        for score in scores:
            one_box = BoundingBox()
            one_box.xmin = int(bboxes[0][i])
            one_box.ymin = int(bboxes[1][i])
            one_box.xmax = int(bboxes[2][i])
            one_box.ymax = int(bboxes[3][i])
            one_box.probability = float(score)
            one_box.class_id = cls[i]
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1
        
        return bboxes_msg

    def joy_callback(self, msg):
        self.joy_command = msg
        
    

    def aligned_depth_image_callback(self, depthImage:Image):
        # size of image is (480, 640)
        robot_stream_depth = self.bridge.imgmsg_to_cv2(depthImage, desired_encoding="passthrough")
        self.robot_stream_depth = np.array(robot_stream_depth, dtype=np.uint16) * 0.001


    def image_callback(self, image:Image):
        image_raw = self.bridge.imgmsg_to_cv2(image, "bgr8")
        # return (class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list, persons = self.yolov5.image_callback(image_raw)

        # if len(persons) > 0:
        #     self.get_logger.info("TARGET POSITION: ", self.target)
            
        # msg = self.yolovFive2bboxes_msgs(bboxes=[x_min_list, y_min_list, x_max_list, y_max_list], scores=confidence_list, cls=class_list, img_header=image.header)

        # self.pub_bbox.publish(msg)

        # self.pub_image.publish(image)
        

        t=Twist()
        v,w = self.get_cmd_vel(persons)
        t.linear.x = float(v/10)
        t.angular.z=float(w)
        self.twist_pub.publish(t)


        print("start ==================")
        print(class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
        print("end ====================")

    def get_position(self,tracked_bbox):
        '''
        Get the distance to the person 
        Removes outliers and get the average distance to 
        the mid point of the bounding box

        Params:
        --------
        tracked box : Top left corner of bounding box,Bottom right corner of bounding box

        Returns:
        -------
        position [x,y,z] the 3D coordinates of pixels p1,p2 the 3D points in camera frame 
        
        '''
        if self.robot_stream_depth is None:
            return (-1, -1, -1)

        #Clipping bbox cooridinates outside the frame
        l= (self.robot_stream_depth.shape)[0]
        w= (self.robot_stream_depth.shape)[1]
        x1=np.clip(tracked_bbox[0],0,w-1)
        x2=np.clip(tracked_bbox[2],0,w-1)
        y1=np.clip(tracked_bbox[1],0,l-1)
        y2=np.clip(tracked_bbox[3],0,l-1)
        x= int((x1+x2)/2)
        y= int((y1+y2)/2)       
        depth=self.robot_stream_depth[y,x]             
        pixel_point=[float(x),float(y)]
        position=rs.rs2_deproject_pixel_to_point(self.depth_intrinsic ,pixel_point,depth)
       
        return position


    def bb_intersection_over_union(self,boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def PID_with_bbox(self):
        target_bbox = self.target
        x1,y1,x2,y2 = target_bbox
        H = y2-y1
        W = x2-x1
        rot_gain = 1/200
        I_gain = 0.5
        P_gain = 0.02
        D_gain = 1
        # PID control basred on the image plane
        # Make sure the image size is 640x480 
        centroid = self.get_centroid(target_bbox)
        # 
        dx = centroid[0]-320 # + -> right - -> left # (pixel -> varies from 50 to 150 pixels)
        dy = centroid[1]-240 # + -> fwd - -> backward
        z = self.target_z # depth to the parameter (in mm)
        
        realx=0.
        if self.xyz is not None:
            realx = self.xyz[0]
        if z<0.5  : # target too close -> stop and rotate little bit towards target
            # bluetooth speaker output
            if self.count == 0: 
                # subprocess.run(["espeak","I am too close to you. I will slow down"])
                self.count += 1
            v=0.0
            w=0.
            
            
        else:
            self.count = 0
            if H>450: # too close
                size_gain=-2
            else:
                size_gain = (H-450)*0.02
            v= I_gain*z + P_gain*dy 
            if dx>50:
                w = -1*rot_gain*dx 
            else:
                w=0.
        
        v = 100*v
        w = 100*w
        if v>1.0:
            v=1.0
        elif v<-1.0:
            v=-1.0
        if w>0.5:
            w=0.5
        elif w<-0.5:
            w=-0.5
        
        return v,w
        
    def get_centroid(self,bbox):
        return [0.5*(bbox[0]+bbox[2]),0.5*(bbox[1]+bbox[3])]
    def get_closest(self,persons):
        Area=[]
        for p in persons:
                
                x1,y1,x2,y2 = p
                Ai = float((x2-x1)*(y2-y1))
                Area.append(Ai)
                
            
        target_i = Area.index(max(Area))
        
        self.target = persons[target_i] 
        
    def get_cmd_vel(self,persons):
        self.view = self.get_position([10,10,630,470])
        # from the previous frame, mention your target person (by bbox)\
        # get the closest one (by centroid)
        if not self.start_process:
            v=0.
            w=0.
            # if self.joy_command is not None and self.joy_command.buttons[7]==1.0: # when the R1 button has pressed in the joystick command
        else:
            
            
            if len(persons)>0:
                if self.target is None:
                    self.get_closest(persons)   
                candidates=[]
                for p in persons :
                    # based on the target IOU        
                    iou_p = self.bb_intersection_over_union(p,self.target)
                    candidates.append(iou_p)
                    
                # person track by best IOU
                idx = candidates.index(max(candidates))
                self.target = persons[idx]
                xyz = self.get_position(self.target) # xyz 3D world coordinate of target
                
                
                self.xyz = xyz
                self.target_z = xyz[2]
                v,w = self.PID_with_bbox() # with target person -> assign v, and w
            elif self.view is not None and self.view[2]<0.5:
                v=-1
                w=0.5 # stop and rotate to find target
             
            elif self.view is not None:
                v=0.0
                w=0.0
                
            else:
                # No person detected
                v=0.0
                w=0.0
    
            
        return v,w
        
    
    
    

def ros_main(args=None):
    rclpy.init(args=args)
    yolov5_node = yolov5_ros()
    rclpy.spin(yolov5_node)
    yolov5_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    ros_main()
