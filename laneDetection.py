from io import BytesIO
from unittest.mock import patch
from cv2 import COLOR_BGR2GRAY, VideoCapture, threshold
import numpy as np
import cv2
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Image
import sys

from scipy import signal
from IPython.display import HTML


from multiprocessing import Pool, pool
from functools import partial

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


class Lane():
    def __init__(self):
        self.left_fitx = None
        self.left_fit = None
        self.right_fitx = None
        self.right_fit = None
        self.ploty = None
        self.detected = False
        self.left_curverad = 0
        self.right_curverad = 0
        self.avg_distance = None
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.out_img=None
     
    def find_lane_pixels(self, binary_warped):
        
        #binary image
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)  
        
        
def merge_image(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

        ##################################################
def process_image(self, img):
########### PUT FULL FUNCTION
    
    
        #find offset
        midpoint = np.int(undist.shape[1]/2)
        middle_of_lane = (self.right_fitx[-1] - self.left_fitx[-1]) / 2.0 + self.left_fitx[-1]
        offset = (midpoint - middle_of_lane) * self.xm_per_pix
        
        resized1 = cv2.resize(w, (200,200), interpolation = cv2.INTER_AREA)
        resized2 = cv2.resize(undist, (200,200), interpolation = cv2.INTER_AREA)
        resized3 = cv2.resize(warped, (200,200), interpolation = cv2.INTER_AREA)
        resized3 = cv2.merge([resized3*255,resized3*255,resized3*255])
        resized4 = cv2.resize(self.out_img, (200,200), interpolation = cv2.INTER_AREA)

        if len(sys.argv) > 1:
            if sys.argv[1] == "debug":
                result = merge_image(result, resized2, 500,25)
                result = merge_image(result, resized1, 700,25)
                result = merge_image(result, resized3, 900,25)
                result = merge_image(result, resized4, 1100,25)
        
        # Add radius and offset calculations to top of video
        cv2.putText(result,"L. Lane Radius: " + "{:0.2f}".format(self.left_curverad/1000) + 'km', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
        cv2.putText(result,"R. Lane Radius: " + "{:0.2f}".format(self.right_curverad/1000) + 'km', org=(50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
        cv2.putText(result,"C. Position: " + "{:0.2f}".format(offset) + 'm', org=(50,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
        return result

def main():

    # initialize a lane object
    calib_params = calibrate_camera()
    lane = Lane()

    # perform the video test
    white_output = 'project1_video_output.mp4'
    if len(sys.argv) > 2:
        path = sys.argv[2]
    else:
        path = "Project_data/challenge_video.mp4"

    if len(sys.argv) > 3:
        white_output = sys.argv[3]
    else:
        white_output = "Project_data/output.mp4"
    clip1 = VideoFileClip(path)
    white_clip = clip1.fl_image(lane.process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    HTML("""
    <video width="860" height="540" controls>
    <source src="{0}">
    </video>
    """.format(white_output))
    print ("................................")

if __name__ == "__main__":
   main()
