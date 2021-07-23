import argparse
import os
import shutil
import time
from pathlib import Path
import math

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from heatmap import Heatmap
class FlowMonitor(object):
    def __init__(self, source, 
                       weights,
                       imgsz=640,
                       conf_thresh=.25,
                       iou_thresh=.45,
                       device='0',
                       use_counter=False, 
                       use_distance=False, 
                       use_heatmap=False):

        view_img = True

        # Initialize
        set_logging()
        device = select_device(device)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        is_stream, dataset = self.createDataloader(source, imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img) if device.type != 'cpu' else None  # run once
        
        first_pass = True

        heatmap = Heatmap([imgsz]*2)

        self.model = model
        self.names = names
        self.colors = colors
        self.is_stream = is_stream
        self.dataset = dataset
        self.heatmap = heatmap
        self.view_img = view_img

        self.source = source # Cross thread control variable
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.use_counter = use_counter # Cross thread control flag
        self.use_distance = use_distance  # Cross thread control flag
        self.use_heatmap = use_heatmap # Cross thread control flag

    def createDataloader(self, source, imgsz):
        is_stream = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
        # Set Dataloader
        if is_stream:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            dataset = LoadImages(source, img_size=imgsz)
            
        return is_stream, iter(dataset)

    def is_close(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        """
        #================================================================
        # 1. Purpose : Calculate Euclidean Distance between two points
        #================================================================    
        :param:
        p1, p2 = two points for calculating Euclidean Distance

        :return:
        dst = Euclidean Distance between two 2d points
        """
        #=================================================================#
        return math.sqrt((x1-x2)**2 + (y1-y2)**2) 

    def updateDataset(self):
        is_stream, dataset = self.createDataloader(self.source, self.imgsz)
        self.is_stream = is_stream
        self.dataset = dataset

    def detect(self):

        try:
            path, img, im0s, vid_cap = next(self.dataset)
        except StopIteration:
            self.updateDataset()
            path, img, im0s, vid_cap = next(self.dataset)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=[0], agnostic=False)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.is_stream:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                
                centers_bbs = []
            
                if self.use_counter:
                    text = "People: %s" % str(len(det)) 			
                    location = (10,25)								
                    im0 = cv2.putText(im0, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x_center = int(xyxy[0]+(xyxy[2]-xyxy[0])/2.0)
                    y_center = int(xyxy[1]+(xyxy[3]-xyxy[1])/2.0)
                    p1 = (x_center,y_center)
                    centers_bbs.append(p1)

                    if self.view_img:  # Add bbox to image
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)

                if self.use_distance:
                    print("DISTANCE")
                    red_zone_list = []
                    red_line_list = []
                    for p1 in range(len(centers_bbs)-1):
                        for p2 in range(p1+1, len(centers_bbs)):
                            distance = self.is_close(centers_bbs[p1], centers_bbs[p2])
                            if distance >= 125.0: continue

                            if p1 not in red_zone_list:
                                red_zone_list.append(p1)       #  Add Id to a list
                                red_line_list.append(centers_bbs[p1])   #  Add points to the list
                            if p2 not in red_zone_list:
                                red_zone_list.append(p2)		# Same for the second id 
                                red_line_list.append(centers_bbs[p2])

                    for idx, detection in enumerate(reversed(det)):  # dict (1(key):red(value), 2 blue)  idx - key  box - value
                        *box, conf, cls = detection
                        box = [int(coord) for coord in box[:4]]
                        color = (0, 0, 255) if idx in red_zone_list else (0, 255, 0)
                        im0 = cv2.rectangle(im0, box[:2], box[2:], color, 2) # Create Red bounding boxes  #starting point, ending point size of 2
        
                    text = "People at Risk: %s" % str(len(red_zone_list)) 			# Count People at Risk
                    location = (10,55)												# Set the location of the displayed text
                    im0 = cv2.putText(im0, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text

                    for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
                        start_point = red_line_list[check] 
                        end_point = red_line_list[check+1]
                        check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
                        check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
                        if (check_line_x < 75) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
                            im0 = cv2.line(im0, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed.

                self.heatmap.store_detections(det)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

        if self.use_heatmap:
            im0 = self.heatmap.generate_heatmap()
        # # Stream results
        # if self.view_img:
        #     cv2.imshow(p, im0)
        #     if cv2.waitKey(1) == ord('q'):  # q to quit
        #         raise StopIteration

        return im0

        # print('Done. (%.3fs)' % (time.time() - t0))
