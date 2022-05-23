from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import sys

import cv2
import numpy as np
import random as rand

import glob
import time


import joblib

from matplotlib.pyplot import imread

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from helpers import convert
from scipy.ndimage import label
# from featuresourcer import FeatureSourcer


from moviepy.editor import VideoFileClip
from IPython.display import HTML
import functools


sourcer_params = {
  'color_model': 'yuv',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             #
  'number_of_orientations': 11,        # 6 - 12
  'pixels_per_cell': 16,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'do_transform_sqrt': True
}

boundingBoxSize = sourcer_params['bounding_box_size']

start_frame = imread("Data/vehicles/KITTI_extracted/5364.png")

ppc_N = sourcer_params['pixels_per_cell']



class FeatureSourcer:
  def __init__(self, p, start_frame):
    
    self.color_model = p['color_model']
    self.s = p['bounding_box_size']
    
    self.ori = p['number_of_orientations']
    self.ppc = (p['pixels_per_cell'], p['pixels_per_cell'])
    self.cpb = (p['cells_per_block'], p['cells_per_block']) 
    self.do_sqrt = p['do_transform_sqrt']

    self.ABC_img = None
    self.dims = (None, None, None)
    self.hogA, self.hogB, self.HogC = None, None, None
    self.hogA_img, self.hogB_img, self.hogC = None, None, None
    
    self.RGB_img = start_frame
    self.new_frame(self.RGB_img)

  def hogFn(self, channel):
    features, hog_img = hog(channel, 
                            orientations = self.ori, 
                            pixels_per_cell = self.ppc,
                            cells_per_block = self.cpb, 
                            transform_sqrt = self.do_sqrt,
                            visualize = True, 
                            feature_vector = False)
    return features, hog_img

  def new_frame(self, frame):
    
    self.RGB_img = frame 
    self.ABC_img = convert(frame, src_model= 'rgb', dest_model = self.color_model)
    self.dims = self.RGB_img.shape
    
    self.hogA, self.hogA_img = self.hogFn(self.ABC_img[:, :, 0])
    self.hogB, self.hogB_img = self.hogFn(self.ABC_img[:, :, 1])
    self.hogC, self.hogC_img = self.hogFn(self.ABC_img[:, :, 2])
    
  def slice(self, x_pix, y_pix, w_pix = None, h_pix = None):
        
    x_start, x_end, y_start, y_end = self.pix_to_hog(x_pix, y_pix, h_pix, w_pix)
    
    hogA = self.hogA[y_start: y_end, x_start: x_end].ravel()
    hogB = self.hogB[y_start: y_end, x_start: x_end].ravel()
    hogC = self.hogC[y_start: y_end, x_start: x_end].ravel()
    hog = np.hstack((hogA, hogB, hogC))

    return hog 

  def features(self, frame):
    self.new_frame(frame)
    return self.slice(0, 0, frame.shape[1] , frame.shape[0])######################added *ppc_N

  def visualize(self):
    return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img

  def pix_to_hog(self, x_pix, y_pix, h_pix, w_pix):

    if h_pix is None and w_pix is None: 
      h_pix, w_pix = self.s, self.s
    
    h = h_pix // self.ppc[0]
    w = w_pix // self.ppc[0]
    y_start = y_pix // self.ppc[0]
    x_start = x_pix // self.ppc[0]
    y_end = y_start + h - 1
    x_end = x_start + w - 1
    
    return x_start, x_end, y_start, y_end

sourcer = FeatureSourcer(sourcer_params, start_frame)



def    newFrame(strip):
    sourcer.new_frame(strip)
    return 0

##this function have to return hog features
def hogSliceFn(strip, resizedX , y, boundingBoxSizeX, boundingBoxSizeY):

    # sourcer.features(strip)
    return sourcer.slice( resizedX, y, w_pix = boundingBoxSizeX, h_pix = boundingBoxSizeY)

####Helper functions####

#get the start and the size of each box, and get the end postion for it
def boxBoundaries(box):
    xStart = box[0]
    yStart = box[1]
    xEnd = box[0] + box[2]
    yEnd = box[1] + box[2]
    return xStart , yStart, xEnd, yEnd 

def drawBoxes(frame, boxes, color = (255,0,0),thick= 10):

    #take a copy from the original image to draw on it all the boxes we want to draw
    outImage = frame.copy()

    #take every box we want to draw on the image,
    #and draw each one of them indvadully
    for box in boxes:

        #get the start position, and the end postion for each traingle
        xStart , yStart, xEnd, yEnd = boxBoundaries(box)

        #draw the rectangle on the image
        cv2.rectangle(outImage, (xStart , yStart),(xEnd, yEnd), color, thick)

    return outImage

#show images function
def showImages(images,nrows,ncols, width, height, depth = 80):
    fig, ax = plt.subplots(nrows= nrows, ncols=ncols, figsize= (width,height),dpi = depth)
    ax = ax.ravel()

    for i in range(len(images)):
        img= images[i]
        ax[i].imshow(img)

    for i in range(nrows*ncols):
        ax[i].axis('off')


##scaling features for training 
def scalingFn(vehiclesFeatures, nonVehiclesFeatures,totalVehicles,totalNonVehicles, pathSaveModel = 'scaler.pkl'):
    ##put all features vectors beside eachothers
    unscaledX = np.vstack((vehiclesFeatures, nonVehiclesFeatures)).astype(np.float64)
    
    #In SVM, data normalization is required to use all kernels related to distance calculation.
    ##standardizion (mean = 0, std = 1) the data which by centerized it and divide by standaard division. betwise the normalize it (the data range bet 0 to 1)
    scalerM = StandardScaler() #creating the module for the standardization
    scaler = scalerM.fit(unscaledX) #Compute the mean and standard devision to be used for later scaling.
    x = scaler.transform(unscaledX) #Perform standardization by centering and scaling.
    y= np.hstack((np.ones(totalVehicles),np.zeros(totalNonVehicles)))


    print ("Saving models...")
    #save the scaler model
    joblib.dump(scaler,pathSaveModel )
    print ("Saving models...")

    return x, y, scaler

##training the data 

def trainFn(xScaledFeatured,yScaledFeatured, pathSaveModel = 'svc.pkl'):

    #split the data bet the training which 70% vs the test wich is 30%, also the chosen data are randomly by precent (rand_state)
    xTrain, xTest, yTrain, yTest = train_test_split(xScaledFeatured,yScaledFeatured, test_size = 0.3,  random_state =rand.randint(1,100))
    
    # SVM is used for both classification and regression problems.
    # Scikit-learn's method of Support Vector Classification (SVC) can be extended to solve regression problems as well. 
    # That extended method is called Support Vector Regression (SVR).
    
    #creating the model
    #svm with linear (also line spearate between the samples ) kernal not segmoid or tanh or relu, between the bars of neural nodes
    svc = LinearSVC() 

    #traning the model with the traintng data data
    svc.fit(xTrain,yTrain)

    #check the accuracy of the model by the testing data
    accuracy = svc.score(xTest,yTest)
    
    print ("Saving models...")
    joblib.dump(svc, pathSaveModel)
    print("...Done")
  
    return svc , accuracy


##predict func 

def predictFn(frameFeature, svc, ScalerModel):
    
    #scale (standrize) the feature, which i use to predict, 
    frameScaled = ScalerModel.transform([frameFeature])

    #make the prediction
    frameClass = svc.predict(frameScaled)

    #return the frame class as int
    return np.int(frameClass[0]) 


##slider : we choose only part of the frame which we call strip to extract the hog feature from it,
#   then predic it every slice on it

#prepare the part we work on to extract the feature, and classify each half cel on it
def prepareSlider(frame , yStart,windowSize,boundingBoxSize):
    
    scaler = windowSize / boundingBoxSize ##for ex 80 /64
    
    #for the strip we get the end
    yEnd = yStart + windowSize

    #the width of the strip which we use it to resize the the stip
    #which will be smaller than the fram width frame by (boundaryBox(64) / windowSize(80) ) 
    #and equal to mulible of boundary boxes by (widthFrame(1024) / windowSize(80)) ~ 12.55
    newWidth = np.int(frame.shape[1] / scaler) # width = boundaryBox(64) * (widthFrame / windowSize(80))

    #take the required strip from the frame
    strip = frame[yStart : yEnd, :]

    #resize this strip
    strip = cv2.resize(strip,(newWidth,boundingBoxSize ))

    return strip, scaler


#to locate the vehicles from the frame
def locateVehicle(frame, yWindowPosition, windowSize, boundingBoxSize, inc, svcModelPath = 'svc.pkl',ScalarModelPath = 'scaler.pkl') :
    
    #load the classifier model
    svc =  joblib.load(svcModelPath)
    ScalarModel = joblib.load(ScalarModelPath)

    #prepare the part we work on to extract the feature, and classify each half cel on it
    strip , scaler = prepareSlider(frame , yWindowPosition,windowSize,boundingBoxSize)

    #boxes which we found the car on it, and want to draw pox on it
    boxes = []

    #make hog to every channel of the strip , in the new frame
    newFrame(strip)

    #just count number of the boxes (blocks) in the width of the strip and substract one from it
    #by taking the floor (stripWidth / boundingBoxSize) => 12 to there no be fractions, then multyply it on  boundingBoxSize => 12 * 64
    # the substract one boundingBoxSize from it => 11 * 64
    #which will be the last x we extract the hog and classfy it 
    xEnd = (strip.shape[1] // boundingBoxSize - 1) * boundingBoxSize

    for resizedX in range(0, xEnd, inc): #if inc = 8 #0,8,16,24, 32, 40, 56, 64, 72, ......

        #extract the features from one slice from the the strip
        # if resizedX = 16 , then iside xStart = floor( 16/pixelpercell) = 1 , width = floor(boundingBoxSize / pexelPerCell) =64/16 = 4, xend = 1 + 4 -1  = 4
        #yStart = 0 / pixelPerCell = 0, height =boundingBoxSize / pexelPerCell  = 4 , yend = 0 + 4 
        #get the feature vector which compines from the 3 channels put besides each other
            #yStartof the real frame  which is the same of the y of window box(yWindowPosition) 
            #because we don't move from the start of the strip = 0 which is represent the yWindowPosition #so the inY here is  = 0
            ### we go through only the top of the strip because if that slice is a part of vehicle, so the whole window is a vehicle 
            #####also maybe if we get part of the car, it will be a car for all the window postion
        features = hogSliceFn(strip, resizedX , 0, boundingBoxSize  , boundingBoxSize)


        #after extraction 
        #we classify (predict)
        #if it's a car we get the xStart position of it in the real fram image not the strip
        if predictFn(features, svc, ScalarModel):

            #xStart = resizedX of the strip* (windowSize / boundingBoxSize) ##for ex 16 *( 80 /64)
            #because we divide the width of the frame by the scaler, so we multyply it again
            xStartV = np.int(scaler * resizedX) 

            #we append every thing about that car, like xStart, yStart which is the same of the y of window box 
            #because we don't move from the start of the strip = 0 which is represent the yWindowPosition
            #also maybe if we get part of the car, it will be a car for all the window postion
            boxes.append((xStartV, yWindowPosition, windowSize))


    return boxes, strip



#get all copies of the images 
def slider(frame,  windowSizes, stripPostions , boundingBoxSize, inc, svcModelPath = 'svc.pkl' , ScalarModelPath = 'scaler.pkl'):

    boxedImages = []
    strips = []
    boungingBoxesTotal = [] 
    ##we go through differnt strip postions on the original frame 410, 390, 380, 380
    # to see if there is a vehicles close to the my car or far away
    #also we got through differen windows because
    #maybe there different size of the vichels 

    ######## the main reason ###################
    # because we classify only the top of the stip 
    # so we can deal if it's a shadow (false positive) or a car (true positive) so we classify the same vehicle from diffiernt postion
    for yWindowPosition, windowSize in zip(stripPostions, windowSizes):

        # Get the vehicles boxes (xStart, yStart, windowsSize) of the original frame from the first strip , second, third, and forth
        boundingBoxes, strip = locateVehicle(frame, yWindowPosition, windowSize, boundingBoxSize, inc, svcModelPath,ScalarModelPath)

        for boundingBox in boundingBoxes:#################################added#########################
            boungingBoxesTotal.append(boundingBox)

        #draw rectangles on each vehicle (of that strip), on a copy from the same original frame, no change for the frame
        boxedImage = drawBoxes(frame, boundingBoxes)

        #collect these copies of the images for the same original frame
        boxedImages.append(boxedImage)

        #collect the strips we work on together
        strips.append(strip)


    return boungingBoxesTotal, boxedImages, strips
    

def HeatmapThresh(heatmap, threshold=1):
    res = np.zeros(heatmap.shape)
    res[heatmap > threshold] = 1
    return res

def HeatmapDraw(thresh_map, frame, color=(0,255,0), thickness=10):
    labeled, num_labels = label(thresh_map)
    for i in range(1, num_labels + 1):
        xs, ys = (labeled == i).nonzero()
        p1 = (np.min(ys), np.min(xs))
        p2 = (np.max(ys), np.max(xs))
        cv2.rectangle(frame, p1, p2, color, thickness)
    return frame

def HeatmapCord_Draw( frame, thresh_map,color=(0,255,0), thickness=10):
    labeled, num_labels = label(thresh_map)
    p = []
    for i in range(1, num_labels + 1):
        xs, ys = (labeled == i).nonzero()
        p1 = (np.min(ys), np.min(xs))
        p2 = (np.max(ys), np.max(xs))
        p.append(( p1 , p2 ))
        cv2.rectangle(frame, p1, p2, color, thickness)

    return frame, p, labeled

def HeatmapAdd(heatmap, pos_x, pos_y, win_size):
        heatmap[pos_y : (pos_y + win_size), pos_x : (pos_x + win_size)] += 1
        return heatmap

def HeatmapUpdate(heatmap, boxes):
    for b in boxes:
        x, y, size = b
        HeatmapAdd(heatmap, x, y, size)
    return heatmap


def verBoseFn(frame,  windowSizes, stripPostions , boundingBoxSize, inc, threshold, svcModelPath = 'svc.pkl' , ScalarModelPath = 'scaler.pkl'):

    boxedImages = []
    strips = []
    boungingBoxesTotal = [] 
    heat = np.zeros(frame.shape[0:2])
    thresh_map = np.zeros(heat.shape)
    ##we go through differnt strip postions on the original frame 410, 390, 380, 380
    # to see if there is a vehicles close to the my car or far away
    #also we got through differen windows because
    #maybe there different size of the vichels 

    ######## the main reason ###################
    # because we classify only the top of the stip 
    # so we can deal if it's a shadow (false positive) or a car (true positive) so we classify the same vehicle from diffiernt postion
    for yWindowPosition, windowSize in zip(stripPostions, windowSizes):

        # Get the vehicles boxes (xStart, yStart, windowsSize) of the original frame from the first strip , second, third, and forth
        boundingBoxes, strip = locateVehicle(frame, yWindowPosition, windowSize, boundingBoxSize, inc, svcModelPath,ScalarModelPath)

        heat = HeatmapUpdate(heat, boundingBoxes)
    

    # plt.subplot(1,2,1)
    # plt.imshow(heat, cmap='hot')

    # plt.subplot(1,2,2)
    # plt.imshow(hm.thresh_map,  cmap='gray')

    # test_frame = np.zeros((720, 1280, 3))
    thresh_map = HeatmapThresh(heat, threshold=threshold)
    frameOut = HeatmapDraw(thresh_map, frame)
    # plt.imshow(frameOut)

    return frameOut

def verBoseEdited (frame):
    # windowSizes = 180, 100, 120, 140, 180, 210
    # stripPostions = 360, 390, 390, 390, 390, 390
    windowSizes = 180, 100, 120, 140 #80,96,160,192
    stripPostions = 360, 390, 390, 390 #
    inc = 8
    threshold = 1

    frameOut =verBoseFn(frame,  windowSizes, stripPostions , boundingBoxSize, inc, threshold, svcModelPath = 'svc.pkl' , ScalarModelPath = 'scaler.pkl')
    return frameOut

# def draw_debug_board(img0, bboxes, hot_windows, heatmap, labels):

def draw_debug_board(img0, hot_windows, heatmap, threshold):
    
    img1 = np.copy(img0)

    img = np.copy(img0)

    thresh_map = HeatmapThresh(heatmap, threshold=threshold)
    img ,posMinMax , labels =HeatmapCord_Draw(img1,thresh_map)
    # plt.imshow(frameOut)
    bboxes = posMinMax
    # hot_windows = boungingBoxesTotal
    
    
    # prepare RGB heatmap image from float32 heatmap channel
    img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8);
    img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_HOT)
    img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

    # prepare RGB labels image from float32 labels channel
    img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8);
    img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)
    
    # draw hot_windows in the frame
    img_hot_windows = np.copy(img)
    img_hot_windows = drawBoxes(img_hot_windows, hot_windows, thick=2)
    
    ymax = 0
    
    board_x = 5
    board_y = 5
    board_ratio = (img.shape[0] - 3*board_x)//3 / img.shape[0] #0.25
    board_h = int(img.shape[0] * board_ratio)
    board_w = int(img.shape[1] * board_ratio)
        
    ymin = board_y
    ymax = board_h + board_y
    xmin = board_x
    xmax = board_x + board_w

    offset_x = board_x + board_w

    # draw hot_windows in the frame
    img_hot_windows = cv2.resize(img_hot_windows, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_hot_windows
    
    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_heatmap = cv2.resize(img_heatmap, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_heatmap
    
    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_labels = cv2.resize(img_labels, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_labels
    
    return img

def verBoseFnDebug(frame,  windowSizes, stripPostions , boundingBoxSize, inc, threshold, svcModelPath = 'svc.pkl' , ScalarModelPath = 'scaler.pkl'):

    boxedImages = []
    strips = []
    boungingBoxesTotal = [] 
    heat = np.zeros(frame.shape[0:2])
    thresh_map = np.zeros(heat.shape)
    ##we go through differnt strip postions on the original frame 410, 390, 380, 380
    # to see if there is a vehicles close to the my car or far away
    #also we got through differen windows because
    #maybe there different size of the vichels 

    ######## the main reason ###################
    # because we classify only the top of the stip 
    # so we can deal if it's a shadow (false positive) or a car (true positive) so we classify the same vehicle from diffiernt postion
    for yWindowPosition, windowSize in zip(stripPostions, windowSizes):

        # Get the vehicles boxes (xStart, yStart, windowsSize) of the original frame from the first strip , second, third, and forth
        boundingBoxes, strip = locateVehicle(frame, yWindowPosition, windowSize, boundingBoxSize, inc, svcModelPath,ScalarModelPath)
        
        for  boundingBox in boundingBoxes:
            boungingBoxesTotal.append(boundingBox)    
        # boungingBoxesTotal.append(boundingBox for boundingBox in boundingBoxes)

        heat = HeatmapUpdate(heat, boundingBoxes)
    

    # plt.subplot(1,2,1)
    # plt.imshow(heat, cmap='hot')

    # plt.subplot(1,2,2)
    # plt.imshow(hm.thresh_map,  cmap='gray')

    # test_frame = np.zeros((720, 1280, 3))
    # thresh_map = HeatmapThresh(heat, threshold=threshold)
    # posMinMax , labeled =HeatmapCord(thresh_map)
    # # plt.imshow(frameOut)
    # bboxes = posMinMax
    hot_windows = boungingBoxesTotal
    
    # hot_windows = [((boundingBox[0]+boundingBox[2]),(boundingBox[1]+boundingBox[2])) for boundingBox in boungingBoxesTotal] 

    frameOut = draw_debug_board(frame, hot_windows, heat, threshold)

    return frameOut


def verBoseEditedDebug (frame):
    # windowSizes = 180, 100, 120, 140, 180, 210
    # stripPostions = 360, 390, 390, 390, 390, 390
    # windowSizes = 180, 100, 120, 140 #80,96,160,192
    windowSizes = 180, 100, 120, 140
    stripPostions = 360, 390, 390, 390
    inc = 8
    threshold = 1

    frameOut =verBoseFnDebug(frame,  windowSizes, stripPostions , boundingBoxSize, inc, threshold, svcModelPath = 'svc.pkl' , ScalarModelPath = 'scaler.pkl')
    return frameOut



def main():

    ##################with debug mode################

    # heatmap = HeatMap( thresh_val=  1)

    deubug = 1
    # inputVideoPath ='Project_data/project_video.mp4'
    # inputVideoPath ='Project_Data/challenge_video.mp4'
    # inputVideoPath = 'Project_Data/harder_challenge_video.mp4'

    # projectOutput = 'Project_data/project_video_output.mp4'

    if len(sys.argv) > 1:
        inputVideoPath = sys.argv[1]
    else:
        inputVideoPath = "Project_data/project_video.mp4"

    if len(sys.argv) > 2:
        projectOutput = sys.argv[2]
    else:
        projectOutput = "Project_data/output.mp4"

    if len(sys.argv) > 3:
        deubug = int(sys.argv[3])
    else:
        deubug = 1



    clip1 = VideoFileClip(inputVideoPath)
    # clip1 = VideoFileClip('Project_Data/challenge_video.mp4')
    # clip1 = VideoFileClip('Project_Data/harder_challenge_video.mp4')


    # verBosePartial = functools.partial(verBose,windowSizes = windowSizes, stripPostions =stripPostions , boundingBoxSize = boundingBoxSize, inc = 8 )
    # whiteClip = clip1.fl_image(verBosePartial)


    if deubug != 1:
            whiteClip = clip1.fl_image(verBoseEdited)
    else :
            whiteClip = clip1.fl_image(verBoseEditedDebug)



    whiteClip.write_videofile(projectOutput,audio= False)
            # threads=5, 


    HTML("""
    <video width="960" height="540" controls>
    <source src="{0}">
    </video>
    """.format(projectOutput))



if __name__=="__main__":
    main()

