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
import pickle

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def binary_warper(img):
    # These are the arrays I calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # undist = undistort_camera(img, calib_params)   
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # create src and dst matrices for perspective transformation 
    src = np.float32([[(0, img_size[1]), (490, 480), (820, 480), (img_size[0],img_size[1])]])
    dst = np.float32([[(0,img_size[1]), (0, 0), (img_size[0], 0), (img_size[0],img_size[1])]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    
    intense = np.sum(warped)

    # binary thresholding with LAB color space using B channel
    lab = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]
    if intense > 400012823:
        b_thresh_min = 230
        b_thresh_max = 250
    elif intense < 110012823:
        b_thresh_min = 70
        b_thresh_max = 150
    else : 
        b_thresh_min = 140
        b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
        
    # binary thresholding with LUV color space using L channel
    luv = cv2.cvtColor(warped, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    if intense > 390012823:
        l_thresh_min = 230
        l_thresh_max = 250
    elif intense < 110012823:
        l_thresh_min = 110
        l_thresh_max = 180
    else :   
        l_thresh_min = 190
        l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(b_binary)
    combined_binary[(b_binary == 1) | (l_binary == 1)] = 1
        
    return combined_binary, Minv, undist, warped 

#######Some test for functions of camera cal, and binary threshol 
# def calibrate_camera():
#     objpoints = []
#     imgpoints = []

#     obj = np.zeros(shape=(6*9, 3), dtype=np.float32)
#     obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)        

#     images = glob.glob("camera_cal/*.jpg")
#     for i in images:
#         img = cv2.imread(i)
#         if(img.shape[0] != 720 or img.shape[1] != 1280):
#             img = cv2.resize(img, (1280, 720))

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.findChessboardCorners(gray, (9, 6))

#         if(ret):
#             imgpoints.append(corners)
#             objpoints.append(obj)

#     return cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)

# def undistort_camera(input_image, calib_params):
#     return cv2.undistort(input_image, calib_params[1], calib_params[2])
# warped = undist[400:,:,:]
# orig = np.float32(  [[560, 50],\
#                     [0, 280],\
#                     [720, 50],\
#                     [1280, 280]])

# dest = np.float32(  [[0,0],\
#                     [0, 300],\
#                     [200, 0],\
#                     [200, 300]])
# \
# def warp(img, orig, dest):
#     trans = cv2.getPerspectiveTransform(orig,  dest)
#     return cv2.warpPerspective(img, trans, (200, 300))
# def threshold(img):
#     res = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     res = res[:,:,2]
#     # res = cv2.medianBlur(res, 5)
#     res = cv2.GaussianBlur(res, (5,1), 0)
#     res = cv2.Sobel(res, cv2.CV_8U, dx=1, dy=0, ksize=3)
#     _, res = cv2.threshold(res, 30, 255, cv2.THRESH_BINARY)
# import imp
# from io import BytesIO
# from cv2 import COLOR_BGR2GRAY, VideoCapture, threshold
# import numpy as np
# import cv2
# import glob
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from IPython.display import Image

# from scipy import signal
# from IPython.display import HTML


# from multiprocessing import Pool, pool
# from functools import partial

# def binaryFilterRoadPavement (img):
#     HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     #calc the dimionsion of rio, and the posion of it
#     width = img.shape[1]
#     height = img.shape[0]
#     xCenter = np.int(width/2)
#     rioWidth = 100
#     rioHeight = 200
#     xLeft  = xCenter  - np.int(rioWidth/2)
#     xRight  = xCenter  + np.int(rioWidth/2)
#     yTop =  height - 30
#     yBottom = height - rioHeight
#     yBottomSmall = yTop - np.int(rioHeight/2)
#     xOffset= 50
#     xFinish =  width - xOffset

#     #get the rio from image by slicing the img
#     #the middle
#     rioCenter = img[yBottom :yTop,xLeft:xRight ]
#     #the left side
#     rioLeft  = img[yBottomSmall:yTop, xOffset:xOffset + rioWidth]
#     #the right side
#     rioRight = img [yBottomSmall:yTop, xFinish- rioWidth : xFinish]
#     #put them beside each other
#     rio = np.hstack(rioCenter, np.vstack(rioLeft,rioRight))##################
#     rioHSV = cv2.cvtColor(rio,cv2.COLOR_BGR2HSV)

#     #calc histiogrma
#     rioHist = cv2.calcHist([rioHSV],[0,1],None, [256,256],[0,256,0,256])


#     #normalize the histo
#     cv2.normalize(rioHist,rioHist,0,255,cv2.NORM_MINMAX)
#     #reflect it into to the img 
#     dst = cv2.calcBackProject([rioHSV],[0,1],rioHist,[0,256,0,256],1)##########

#     #cracte the filter we go with it throught the image
#     disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     cv2.filter2D(dst, -1, disc, dst)

#     #thershold the image
#     ret , thersh  = cv2.threshold(dst,10, 250, cv2.THRESH_BINARY_INV)

#     return thersh


# def compinedThreshold(img, kernal = 3, gradThreshold= (30,100), magThreshold =(70, 100),
#                     dirThreshold =(0.8, 0.9), sThreshold = (100, 255),rThreshold = (150, 255),
#                     uThreshold = (140,180), threshold = "dayTime-filter-pavement"):


#     def binaryThreshold(channel, thresh =(200,255), on = 1):
#         binary = np.zeros_like(channel)
#         binary[(channel > thresh[0]) & (channel <= thresh[1])] = on

#         return binary
        
#     if threshold in ["dayTime-bright",  "dayTime-filter-pavement"]:
#         rThreshold = (210, 255)

#     #cvt to gray
#     gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     #sobel in x dir, and y dir
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize= kernal)
#     sobely = cv2.Sobel(gray, cv2.CV_64F , 0, 1, ksize=kernal)

#     #sobel x grad binary
#     absSobelX = np.absolute(sobelx)
#     scaledAbsSobelx = np.uint8(255 * (absSobelX/np.max(absSobelX)))
#     gradX = binaryThreshold(scaledAbsSobelx,gradThreshold)

#     #sobel y grad binary
#     absSobelY = np.absolute(sobely)
#     scaledAbsSobelY = np.uint8(255 * (absSobelY/np.max(absSobelY)))
#     gradY = binaryThreshold(scaledAbsSobelY, gradThreshold)


#     #sobel mag grad binary
#     gradMag = np.sqrt(sobelx*2 + sobely*2)
#     scaledGradMag = np.uint8(255 * (gradMag/ np.max(gradMag)))
#     gradMagBinary = binaryThreshold(scaledGradMag, magThreshold)

#     #sobel dir binar
#     # gradDir = np.arctan2(absSobelY/absSobelY)
#     gradDir = np.arctan2(absSobelY,absSobelX)
#     scaledGradDir = np.uint8(255 * (gradDir / np.max(gradDir)))
#     gradDirBinary =  binaryThreshold(scaledGradDir, dirThreshold)


#     #RGB color
#     R = img[:,:,2]
#     G = img[:,:,1]
#     B = img[:,:,0]
#     RBinary = binaryThreshold(R,rThreshold)

#     #HLS colour 
#     hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#     H = img[:,:,0]
#     L = img[:,:,1]
#     S = img[:,:,2]
#     SBinary = binaryThreshold(S, sThreshold)

#     #YUV colour 
#     hls = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     Y = img[:,:,0]
#     U = img[:,:,1]
#     V = img[:,:,2]
#     UBinary = binaryThreshold(U, uThreshold)


#     compined = np.zeros_like(gradDirBinary)

#     if threshold == "dayTime-normal":
#         compined[ (gradX==1) | (SBinary==1) | (RBinary == 1) ] = 1
#     elif threshold == "dayTime-shadow":
#         compined[((gradX == 1) & (gradY == 1)) | (RBinary == 1)] = 1
#     elif threshold == "dayTime-bright":
#         compined[((gradX == 1) & (gradY ==1)) | ((gradMagBinary == 1)& (gradDirBinary == 1)) | (SBinary == 1) | (RBinary == 1)] = 1
#     elif threshold == "dayTime-filter-pavement":
#         roadBinary = binaryThreshold(binaryFilterRoadPavement(img))
#         compined [(((gradX ==1) | (SBinary ==1)) & (roadBinary == 1)) | (RBinary == 1)] = 1
#     else:
#         compined[((gradX ==1) | (RBinary ==1)) & ((SBinary == 1) | (UBinary == 1) | (RBinary == 1))] = 1

#     return compined
    

    
# def perspective_transforms(src,dst):
#     M = cv2.getPerspectiveTransform(src,dst)
#     MInv = cv2.getPerspectiveTransform(src,dst)
#     return M, MInv

# def warp(img, M):
#     imgSize = (img.shape[0],img.shape[1])

#     warpped = cv2.warpPerspective(img, M,imgSize,flags = cv2.INTER_LINEAR)
#     return warpped

# def uuWarp(img, MInv):
#     imgSize = (img.shape[0],img.shape[1])

#     unWarpped = cv2.warpPerspective(img, MInv,imgSize,flasgs = cv2.INTER_LINEAR)
#     return unWarpped

# def clacWarpPoint(imgHeight, imgWidth, xCenterAdj = 0):

#     imgShape  = (imgHeight,imgWidth)
#     xCenter = imgShape[1]/2 + xCenterAdj

#     xBottom  = 54#fronttt
#     yBottom = 450
#     xOffset = 120

#     src = np.float32(
#         [(xOffset,imgShape[0]),
#         (xCenter-xBottom,yBottom),
#         (xCenter + xBottom, yBottom),
#         (imgShape[1]-xOffset, imgShape[0])

#         ]
#     )

#     #with the same order we map 1st point -> 1st point 

#     dst = np.float32(
#         [(xOffset,imgShape[1]),
#         (xOffset,0),
#         (imgShape[0]-xOffset, 0),
#         (imgShape[0]-xOffset, imgShape[1])
#         ]
#     )

#     return src , dst


# def binaryImage(img, calib_params , threshold ="dayTime-normal"):

    
#     def noiseDetect(warpped) :
#         histogram = np.sum(warpped, axis= 1)
#         return (histogram>100).any() 

#     #############undistort
#     undistort = undistort_camera(img, calib_params)

#     src, dst = clacWarpPoint(undistort.shape[0],undistort.shape[1])

#     M, _ =perspective_transforms(src,dst)

#     if threshold == "dayTime-normal":

#         compined =compinedThreshold(undistort, threshold=threshold)
#         warpped  = warp(compined,M)
#         if noiseDetect(warpped):
#             compined = compinedThreshold(undistort, threshold= "dayTime-shadow")
#             warpped = warp(compined, M)

#     else :
#         compined =compinedThreshold(undistort, threshold=threshold)
#         warpped  = warp(compined,M)

#     return warpped
#######Some tests for the lane detection#########
# def laneHisto(img, heightS=800, heightE = 1250):
#     histogram = np.sum(img[int(heightS):int(heightE),:], axis = 0)
#     return histogram

# def lanePeaks(histo):
#     peaks= signal.find_peaks_cwt(histo, np.arange(1,150),min_length=150)
#     midpoint = np.int(histo.shape[0]/2)


#     #if there more than one peaks happens in shadows
#     if len(peaks) > 1:
#         peakLeft , *_, PeakRight = peaks

#     else :
#         #there's only two
#         peakLeft = np.argmax(histo[:midpoint])
#         peakRight =midpoint + np.argmax(histo[midpoint:])

#     return peakLeft, peakRight


# class WindowBox (object):
#     def __init__(self, binaryimg, xCenter,yTop , width = 100, height = 80, minCount = 50, laneFound = False):
#         self.xCenter = xCenter
#         self.yTop = yTop
#         self.width = width
#         self.height = height
#         self.minCount = minCount
#         self.laneFound = laneFound

#         #drived
#         self.xLeft = self.xCenter - int(self.width/2)
#         self.xRight = self.xCenter + int(self.width/2)
#         self.yBottom = self.yTop - self.height

#         self.imgWindow = binaryimg[
#             self.yBottom:self.yTop,
#             self.xLeft: self.xRight
#         ]

#         #get these non zeros from the slice of the image so 0,0 it's for the window not the total img
#         self.nonZeroY = self.imgWindow.nonzero()[0]
#         self.nonZeroX = self.imgWindow.nonzero()[1] 
#         # self.nonZeroY = np.nonzero(self.imgWindow)[0]
#         # self.nonZeroX = np.nonzero(self.imgWindow)[1]

        
#     def center(self):
#         return (self.xCenter, (self.yTop-self.yBottom)/2)

#     def _nonZeroCount(self):
#         return len(self.nonZeroX)

#     def _isNoise(self):
#         return (self._nonZeroCount()>self.imgWindow.shape[0] * self.imgWindow.shape[1] * 0.75)

#     def hasLine(self):
#         return (self._nonZeroCount() > self.minCount ) ^ self._isNoise() 

#     def nextWindow(self, binaryImg):

#         if self.hasLine():
#             #recenter
#             xCenter  = np.int(np.mean(self.nonZeroX + self.xLeft))

#         else:
#             #same
#             xCenter = self.xCenter

#         #it starts from the one after it 
#         yTop = self.yBottom

#         return WindowBox(binaryImg,xCenter,yTop, width=self.width,
#                         height=self.height, minCount= self.minCount,
#                         laneFound=self.laneFound)

#     def hasLane(self):
#         if not self.laneFound and self.hasLine():
#             self.laneFound =True
#         return self.laneFound

#     def getYTop(self):
#         return self.yTop
#     def __str__(self):
#         return "WindowBox [%.3f, %.3f, %.3f, %.3f]" % (self.xLeft,
#                                                        self.yBottom,
#                                                        self.xRight,
#                                                        self.yTop)


# def findLaneWindows(windowBox, binaryImg):
#     boxes = []

#     #add all windows on the frame for one line

#     continueLaneSearch  = True
#     contugousBoxNoLineCount = 0

#     #we search uless the frame is ended or we go on wrong path where there no lines
#     while(continueLaneSearch and windowBox.getYTop() >= 0):
#         if windowBox.hasLine():
#             boxes.append(windowBox)

#         windowBox = windowBox.nextWindow(binaryImg)

#         if windowBox.hasLane() :
#             if windowBox.hasLine():
#                 contugousBoxNoLineCount = 0
#             else:
#                 contugousBoxNoLineCount += 1

#                 if contugousBoxNoLineCount >= 4:
#                     continueLaneSearch = False

#     return boxes



# def calcLeftAndRightlaneWindows (binaryImage, nWindows = 12 , width = 100):

#     windowHeight = np.int(binaryImage.shape[0] / nWindows)

#     #left, and right lane centers
#     windowStartLeft , windowStartRight = lanePeaks(laneHisto(img=binaryImage))

#     #the y we start from to detect lines
#     windowYTop = binaryImage.shape[0]


#     startWindBoxL = WindowBox(binaryImage, windowStartLeft, windowYTop, width=width, height=windowHeight)

#     startWindBoxR = WindowBox(binaryImage, windowStartRight, windowYTop, width=width, height=windowHeight)
    

#     findLaneWindowsFixedArg = partial(findLaneWindows, binaryImg= binaryImage)

#     # paralel function in two threads with different data
#     with Pool(2) as p : 
#         leftBox, rightBox = p.map(findLaneWindowsFixedArg, [startWindBoxL,startWindBoxR])

#     # leftBox = findLaneWindows(startWindBoxL, binaryImg=binaryImage)
#     # rightBox = findLaneWindows(startWindBoxR, binaryImg=binaryImage)

#     return (leftBox, rightBox)

# def calcFitFromBoxes(boxes):
#     if len(boxes) > 0:
#         #flatten put the vales besides each othe in 1d array

#         #the nonzero represnt the point in the window img not the total image so we sum (0,0) on it left
#         x = np.concatenate([box.nonZeroX + box.xLeft for box in boxes])
#         y = np.concatenate([box.nonZeroY + box.yBottom for box in boxes])

#         return np.polyfit(y,x,2) #a x**2 + b * x + c
#     else:
#         return None



# def clacFitX(fitY, lineFit):
#     fitLineX = lineFit[0] * fitY**2 + lineFit[1] * fitY + lineFit[2]
#     return fitLineX


# def calcLeftRightFromPoly(binimg, leftFit, rightFit, margin):
#     nonzero = binimg.nonzero()
#     nonzeroY = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])

#     def windowLane(poly):
#         return (
#             (nonzerox > (poly[0]* (nonzeroY**2) + poly[1]*nonzeroY + poly[2] - margin))
#             & (nonzerox > (poly[0]* (nonzeroY**2) + poly[1]*nonzeroY + poly[2] + margin))

#         ) 

#     def windowPolyFit (lane):
#         x = nonzerox[lane]
#         y = nonzeroY[lane]

#         return np.polyfit(y,x,2)

#     newLeftFit = leftFit
#     if leftFit is not None:
#         newLeftFit = windowPolyFit(windowLane(leftFit))

#     newRightFit = rightFit

#     print(rightFit)
#     if rightFit is not None:
#         newRightFit = windowPolyFit(windowLane(rightFit))

#     return (newLeftFit,newRightFit)

# def drawLaneWarp(img):
#     height = img.shape[0]
#     leftX, ritghX = lines(img, withMargin = 0)
#     topDowmImage = binaryImage(img)

#     heightTD = topDowmImage[0].shape[0]
#     fitY = np.linspace(0, heightTD-1, heightTD)


#     black = np.zeros_like(topDowmImage[0])

#     pts_left = np.array([np.transpose(np.vstack([leftX,fitY]))])
#     pts_right = np.array([np.flipud(np.transpose(np.vstack([ritghX,fitY ])))])
#     pts = np.hstack((pts_left, pts_right))
   

#     cv2.fillPoly(black, np.int_([pts]),(255,255,255))

#     histogram = laneHisto(topDowmImage[0])
#     hist_pts=[[i, height-histogram[i]] for i in range(0,topDowmImage[0].shape[1] -1)]

#     cv2.polylines(topDowmImage[0],np.int_([hist_pts]), False, (255,255,255))
#     cv2.polylines(black,np.int_([pts_left]), False, (255,255,255))
#     cv2.polylines(black,np.int_([pts_right]), False, (255,255,255))

#     # result = topDowmImage | black


#     result = cv2.addWeighted(topDowmImage[0], .7,black, 0.2, 0)
#     # tt = unWarp(result, topDowmImage[1])

#     plt.imshow(result ,cmap="gray")
#     return result

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
    def measure_curvature(self):
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval*self.ym_per_pix + self.left_fit[1])**2)**1.5) / abs(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval*self.ym_per_pix + self.right_fit[1])**2)**1.5) / abs(2*self.right_fit[0])
        
        return left_curverad, right_curverad
    
    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        # set detected false, only set it true if polynomials are fitted
        self.detected = False
        # return if no left line or right line is detected and use previous fitted polynomials
        if len(leftx) == 0 or len(rightx) == 0: return
         ### Fit a second order polynomial to each with np.polyfit() ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        # cal curvature
        left_curverad, right_curverad = self.measure_curvature()
        
        ### Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        
        # calc average distance
        avg_distance = abs(np.mean(left_fitx) - np.mean(right_fitx))
        if self.avg_distance is not None:
            # means it not first frame
            diff_curv = abs(left_curverad - right_curverad)
            if diff_curv > 4000:
                return # if difference in curvatures is greater than 4 km don't fit polynomials
            distance_diff = abs(avg_distance - self.avg_distance)
            if distance_diff > 100: # if difference in pass and present avg distance is greater than 100 px don't fit polynomials
                return
        self.out_img[lefty, leftx] = [255, 0, 0]
        self.out_img[righty, rightx] = [0, 0, 255]
        self.left_curverad = left_curverad
        self.right_curverad = right_curverad
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.detected = True # line detected
        self.avg_distance = avg_distance
     
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
        
    ##################################################
    def process_image(self, img):
        ########### PUT FULL FUNCTION
        warped, Minv, undist, w = binary_warper(img)
        if self.detected:
            # if lines are detected in past frame use those for fitting new polynomials
            self.search_around_poly(warped)
        else:
            # otherwise calculate new polynomials from scratch
            self.find_lane_pixels(warped)
        # if nothing is fitted then return the undistored image
        if self.left_fitx is None or self.right_fitx is None: return undist
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        #find offset
        midpoint = np.int(undist.shape[1]/2)
        middle_of_lane = (self.right_fitx[-1] - self.left_fitx[-1]) / 2.0 + self.left_fitx[-1]
        offset = (midpoint - middle_of_lane) * self.xm_per_pix
    
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


def main():

    # initialize a lane object
    # calib_params = calibrate_camera()
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
