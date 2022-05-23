
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

print("Loading images to memory...")
t_start = time.time()

vehicle_imgs, nonvehicle_imgs = [], []
vehicle_paths = glob.glob('Data/vehicles/*/*.png')
nonvehicle_paths = glob.glob('Data/non-vehicles/*/*.png')

for path in vehicle_paths: vehicle_imgs.append(imread(path))
for path in nonvehicle_paths: nonvehicle_imgs.append(imread(path))

vehicle_imgs, nonvehicle_imgs = np.asarray(vehicle_imgs), np.asarray(nonvehicle_imgs)
total_vehicles, total_nonvehicles = vehicle_imgs.shape[0], nonvehicle_imgs.shape[0]

print("... Done")
print("Time Taken:", np.round(time.time() - t_start, 2))
print("Vehicle images shape: ", vehicle_imgs.shape)
print("Non-vehicle images shape: ", nonvehicle_imgs.shape)

print("Extracting features... This might take a while...")
t_start = time.time()

vehicles_features, nonvehicles_features = [], []

print("Vehicles...")
for img in vehicle_imgs:
  vehicles_features.append(sourcer.features(img))
  print('█', end = '')

print()
print("Non-Vehicles...")
for img in nonvehicle_imgs:
  nonvehicles_features.append(sourcer.features(img))
  print('█', end = '')
                         
vehicles_features = np.asarray(vehicles_features)
nonvehicles_features = np.asarray(nonvehicles_features)

print()
print("...Done")
print("Time Taken:", np.round(time.time() - t_start, 2))
print("Vehicles features shape: ", vehicles_features.shape)
print("Non-vehicles features shape: ", nonvehicles_features.shape)


##parameters

vehiclesFeatures = vehicles_features 
nonVehiclesFeatures = nonvehicles_features

totalVehicles,totalNonVehicles = total_vehicles,total_nonvehicles

print("Scaling features...")
t_start = time.time()

xScaledFeatured,yScaledFeatured,scaler = scalingFn(vehiclesFeatures, nonVehiclesFeatures,totalVehicles,totalNonVehicles, pathSaveModel = 'scaler.pkl')

print("...Done")
print("Time Taken:", np.round(time.time() - t_start, 2))
print(" x shape: ", xScaledFeatured.shape, " y shape: ", yScaledFeatured.shape)


print("Training classifier...")
t_start = time.time()

svc , accuracy = trainFn(xScaledFeatured,yScaledFeatured, pathSaveModel = 'svc.pkl')



print("...Done")
print("Time Taken:", np.round(time.time() - t_start, 2))
print("Accuracy: ", np.round(accuracy, 4))