import numpy as np 
import sys,os
import cv2
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils import *
import random

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
    '''
    plt.imshow(image)
    plt.colorbar()
    plt.show()
    
    cv2.imwrite('/home/sebasmos/Documentos/NASA_Spacesuit/NASA_SpatiumAESS/Training_code/code/image.jpg',image)
    '''
    return image
# Fast testing bef assignmt
path = "/home/sebasmos/Documentos/NASA_Spacesuit/train/images/933760.jpg"
#segmentation(path, 1)
    

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



def gen_array():
    x_f = []
    y_f = []
    v_f = []

    for i in range(15):
        x_f.append(int (random.uniform(100,999)))
        y_f.append(int (random.uniform(100,999)))
        v_f.append(int (random.uniform(0,3)))

    return x_f,y_f,v_f


def gen_geometries():
    x_f = []
    y_f = []
    type = 0
    if type == 0: #rectangular then 8 var, 4 corners
        for i in range(4):
            x_f.append(int (random.uniform(100,999)))
            y_f.append(int (random.uniform(100,999)))
    elif type==1: # 3 corners
        print("triangular")
    else:
        print("no figure detected")
    return x_f,y_f
    

def generar_csv(imag,coo,polig):

   f=open(local_dir,'w')

   for l in range (len(imag)):
      
      f.write(str(imag[l])+','+str(coo[l])+','+str(polig[l])+'\n')

   f.close()
print("Stored in ../../../solution_sintetica.csv")

