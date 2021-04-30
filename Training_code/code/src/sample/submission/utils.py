import numpy as np 
import sys,os
import cv2
import matplotlib.pyplot as plt
import argparse
import random
import pandas as pd

def segmentation(path, label):

    IMG_SIZE = 500
    image = cv2.imread(path)
    #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
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

    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    # output segmented image with colorbar
    
    rows,cols = image.shape

    # Create mask
    mask = image.copy()

    for i in range(rows):
        for j in range(cols):
            k = image[i,j]
            if k > 100:
                mask[i,j] = 1
            else:
                mask[i,j] = 0
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    '''
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    
    cv2.imwrite('/home/sebasmos/Documentos/NASA_Spacesuit/NASA_SpatiumAESS/Training_code/code/image.jpg',image)
    '''
    return mask


def feature_extraction(path, label): # ADAPT ANNOTATIONS CORRECTLY & POLYGON + AREA EXTRACTION
  #ima = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  
  segm = segmentation(path, label)
  mean = np.mean(segm)
  #contrast = np.amax(ima)-np.amin(ima)
  #sdev = np.std(ima)
  #ssd_blur, blur_count = homogeinty(path)
  #features = np.array([label, mean, contrast, sdev, ssd_blur, blur_count])
  #features = np.array([label, mean])<
  
  return mean


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
    
def generar_csv (imag,coo,polig):

   f=open('../solution/solution.csv','w')

   for l in range (len(imag)):
      
      f.write(str(imag[l])+','+str(coo[l])+','+str(polig[l])+'\n')

   f.close()
