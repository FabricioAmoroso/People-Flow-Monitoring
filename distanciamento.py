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

def is_close(p1, p2):
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
    dst = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    #=================================================================#
    return dst 

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    first_pass = True
    for path, img, im0s, vid_cap in dataset:
        if first_pass:
            img_dims = img.shape[2:4] if len(img.shape) == 4 else img.shape[1:3]
            heatmap = Heatmap(img_dims)
            first_pass = False
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str("testezao.avi")
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                centers_bbs = []
            
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    text = "People: %s" % str(len(det)) 			# Count People at Risk
                    location = (10,25)												# Set the location of the displayed text
                    cv2.putText(im0, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text
                    x_center = int(xyxy[0]+(xyxy[2]-xyxy[0])/2.0)
                    y_center = int(xyxy[1]+(xyxy[3]-xyxy[1])/2.0)
                    p1 = (x_center,y_center)
                    centers_bbs.append(p1)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                red_zone_list = []
                red_line_list = []
                for p1 in range(len(centers_bbs)-1):
                    for p2 in range(p1+1, len(centers_bbs)):
                        distance = is_close(centers_bbs[p1], centers_bbs[p2])
                        if distance < 125.0:
                            if p1 not in red_zone_list:
                                red_zone_list.append(p1)       #  Add Id to a list
                                red_line_list.append(centers_bbs[p1])   #  Add points to the list
                            if p2 not in red_zone_list:
                                red_zone_list.append(p2)		# Same for the second id 
                                red_line_list.append(centers_bbs[p2])
                print(len(red_zone_list))
                for idx, detection in enumerate(reversed(det)):  # dict (1(key):red(value), 2 blue)  idx - key  box - value
                    *box, conf, cls = detection
                    if idx in red_zone_list:   # if id is in red zone list
                        cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2) # Create Red bounding boxes  #starting point, ending point size of 2
                    else:
                        cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) # Create Green bounding boxes
      
                text = "People at Risk: %s" % str(len(red_zone_list)) 			# Count People at Risk
                location = (10,55)												# Set the location of the displayed text
                cv2.putText(im0, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text

                for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
                    start_point = red_line_list[check] 
                    end_point = red_line_list[check+1]
                    check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
                    check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
                    if (check_line_x < 75) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
                        cv2.line(im0, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed.

            heatmap.store_detections(det)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    #vid_writer.write(im0)
                    cv2.imshow("Weber",im0)
                    k = cv2.waitKey(1)
                    if k == ord("q"):
                        exit(0)
                        cv2.destroyAllWindows()

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
