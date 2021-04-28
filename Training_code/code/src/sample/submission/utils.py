import numpy as np 
import sys,os
import cv2
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def segmentation(path, label):
    image = cv2.imread(path)
    mask = np.zeros(image.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    rectangle = (0, 0, 500, image.shape[1])
    cv2.grabCut(image, mask, rectangle,  
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
   
    # The final mask is multiplied with 
    # the input image to give the segmented image.
    image = image * mask2[:, :, np.newaxis]
      
    # output segmented image with colorbar
   
    plt.imshow(image)
    plt.colorbar()
    plt.show()
    '''
    cv2.imwrite('/home/sebasmos/Documentos/NASA_Spacesuit/NASA_SpatiumAESS/Training_code/code/image.jpg',image)
    '''
    return 1
    

def feature_extraction(path, label): # ADAPT ANNOTATIONS CORRECTLY & POLYGON + AREA EXTRACTION
  #ima = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  #mean = np.mean(ima)
  segm = segmentation(path, label)
  #contrast = np.amax(ima)-np.amin(ima)
  #sdev = np.std(ima)
  #ssd_blur, blur_count = homogeinty(path)
  #features = np.array([label, mean, contrast, sdev, ssd_blur, blur_count])
  #features = np.array([label, mean])
  
  return segm

