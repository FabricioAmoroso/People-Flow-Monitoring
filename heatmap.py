import numpy as np
from time import time
import cv2

class Heatmap(object):
    """
    This class takes YOLO detections from a camera, translates to reduced working space, 
    which allows less computing and better generalization (e.g. 1920x1080 -> 640x480). After that,
    those coordinates are saved and kept until a thresholded number of frames is hit, which leds to
    averaging of the saved detections and, posteriorly, construction of the heatmap with
    computed averages.
    """

    def __init__(self, camera_dims):
        """
        Heatmap class constructor

        Parameters
        ----------
            `camera_dims` : tuple
                Two values corresponding respectively to camera width and height

        Returns
        -------
            None
        """
        # Declare global variables
        self.camera_height, self.camera_width = camera_dims
        self.grid_side = max(camera_dims)//8 # grid_side

        self.dets_thresh = 0
        self.last_dets = None

        self.grids_remove_movement_temp = []
        self.grids_remove_movement = np.zeros((self.grid_side, self.grid_side), dtype=np.uint16)

        colorspace_bar = np.array([np.linspace(255,0,self.camera_height,dtype=np.uint8) for _ in range(15)])
        self.colorspace_bar = cv2.applyColorMap(colorspace_bar.transpose(), cv2.COLORMAP_JET)
        self.white_margin = 255*np.ones((self.camera_height,8,3),dtype=np.uint8)

    def _grid_averages(self):
        """
        Average the number of detections from a set.(e.g. takes the average number of people in scene 
        every 3 detections).

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        new_grid = np.zeros((self.grid_side, self.grid_side), dtype=np.uint8)

        temp_grids = self.grids_remove_movement_temp

        for temp_grid in temp_grids:
            new_grid += temp_grid
        new_grid = (new_grid // len(temp_grids))
        self.grids_remove_movement_temp = []

        self.grids_remove_movement += 8*new_grid
        self.grids_remove_movement = np.clip(self.grids_remove_movement, 0, 255)
        self._generate_heatmap()

    def _box_center(self, xyxy):
        """
        Convert bounding box coordinates from top-right/bottom-left to center. (XlYt,XrYb)->(XcYc)

        Parameters
        ----------
            `xyxy`: tuple
                Tuple containing bounding box top-right/bottom-left coordinates for conversion

        Returns
        -------
            `center`: tuple
                Tuple containing Xc,Yc coordinates converted from XlYt,YtYb
        """

        center = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2.0), int(xyxy[1]+(xyxy[3]-xyxy[1])/2.0))
        return center

    
    def store_detections(self, detections):
        """
        Store detections for future averaging.
    
        Parameters
        ----------
            `detections`: list
                Python list containing bounding box coordinates, confidence and class.
    
        Returns
        -------
            None
        """

        self.dets_thresh += 1
        temp_grid = np.zeros((self.grid_side, self.grid_side), dtype = np.uint8)
        
        self.last_dets = []
        if detections is not None and len(detections):
            for *xyxy, _, _ in reversed(detections):
                x, y = self._box_center(xyxy)
                if x >= self.camera_width or y >= self.camera_height:
                    continue

                x_norm = x/float(self.camera_width)  # Coord in range:
                y_norm = y/float(self.camera_height) #     [0,1]

                x_grid = int((x_norm * self.grid_side) // 1) 
                y_grid = int((y_norm * self.grid_side) // 1) 

                temp_grid[y_grid][x_grid] += 1

                self.last_dets.append((y, x))
        self.grids_remove_movement_temp.append(temp_grid)

        if self.dets_thresh >= 5:
            self._grid_averages()   

    def _generate_heatmap(self):
        """
        Draw averaged detections on heatmap.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
    
        z = (self.grids_remove_movement).astype(np.uint8)
        z = cv2.resize(z, (self.camera_width, self.camera_height))
        z = cv2.GaussianBlur(z, (23,23), 0)

        image = z

        image = cv2.applyColorMap(z, cv2.COLORMAP_JET)
        if self.last_dets:
            for det in self.last_dets:
                y, x = det
                cv2.circle(image, (x,y), 5, (0,255,0), -1)

        margin = self.white_margin
        image = np.hstack((margin, image, margin, self.colorspace_bar, margin))

        cv2.imshow('Heatmap', image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit(0)





