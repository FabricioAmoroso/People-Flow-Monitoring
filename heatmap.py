import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

class Heatmap(object):

    def __init__(self, camera_dims, grid_side):
        ### Declare global variables ###
        self.camera_height, self.camera_width = camera_dims
        self.grid_side = grid_side

        self.dets_thresh = 0
        self.first_detection = True

        self.grids_remove_movement_temp = []
        self.grids_remove_movement = np.zeros((self.grid_side, self.grid_side), dtype=np.float32)

    def grid_averages(self):

        new_grid = np.zeros((self.grid_side, self.grid_side), dtype=np.float32)

        temp_grids = self.grids_remove_movement_temp

        for temp_grid in temp_grids:
            new_grid += temp_grid
        new_grid = (new_grid // len(temp_grids))
        self.grids_remove_movement_temp = []

        self.grids_remove_movement += new_grid
        print(self.grids_remove_movement)
        self.generate_heatmap()

    def box_center(self, xyxy):
        center = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2.0), int(xyxy[1]+(xyxy[3]-xyxy[1])/2.0))
        return center

    def store_detections(self, detections):
        if self.first_detection:
            self.time_counter = time()
            self.first_detection = False

        self.dets_thresh += 1
        temp_grid = np.zeros((self.grid_side, self.grid_side), dtype = np.float32)
        
        if detections is not None and len(detections):
            for *xyxy, _, _ in reversed(detections):
                x, y = self.box_center(xyxy)

                x_norm = x/float(self.camera_width)  # Coord in range:
                y_norm = y/float(self.camera_height) #     [0,1]

                x_grid = int((x_norm * self.grid_side) // 1) 
                y_grid = int((y_norm * self.grid_side) // 1) 

                temp_grid[y_grid-1][x_grid-1] += 1
        self.grids_remove_movement_temp.append(temp_grid)

        if self.dets_thresh >= 5:
            self.grid_averages()   

    def generate_heatmap(self):
        z = (np.clip(self.grids_remove_movement,0,255)).astype('uint8')
        z = cv2.resize(z, (640,480))

        cv2.imshow('jonas', z)
        k = cv2.waitKey(1)
        if k == ord('q'):
            exit(0)

        # y, x = np.meshgrid(np.linspace(0, self.grid_side, self.grid_side), np.linspace(0, self.grid_side, self.grid_side))
        # print(y.shape)
        # z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        # z = (z.astype('float32') / (z_max/2.0)) - 1.0

        # z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        # fig, ax = plt.subplots()

        # c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        # ax.set_title('pcolormesh')
        # # set the limits of the plot to the limits of the data
        # ax.axis([x.min(), x.max(), y.min(), y.max()])
        # fig.colorbar(c, ax=ax)

        # plt.savefig('temp.png')
        # cv2.imshow('jonas', cv2.imread('temp.png'))
        # k = cv2.waitKey(1)
        # if k == ord('q'):
        #     exit(0)

        # plt.show()


