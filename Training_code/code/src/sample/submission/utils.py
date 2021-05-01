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
    
    mask = np.zeros(image.shape[:2], np.uint8) # create a similar mask image

    backgroundModel = np.zeros((1, 65), np.float64) # These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays of size (1,65).
    
    foregroundModel = np.zeros((1, 65), np.float64) # These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays of size (1,65).

    rectangle = (50,50,450,290)

    cv2.grabCut(image, mask, rectangle,  
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') # apply mask
    # So we modify the mask such that all 0-pixels and 2-pixels are put to 0 (ie background) and all 1-pixels and 3-pixels are put to 1(ie foreground pixels). Now our final mask is ready. Just multiply it with input image to get the segmented image. 
    #print(mask2)
    img = image*mask2[:,:,np.newaxis]
    
    [rows,cols]= mask2.shape
    
    triplets = []
    for i in range(rows):
        for j in range(cols):
            k = mask2[i,j]
            if k == 1:
                aux = [i,j,int(random.uniform(0,3))]

                triplets.append(aux)
                #print(triplets)
                #mask[i,j] = 1
            else: # ADD HERE LABEL 1 AND LABEL 0
                aux = 0
                #print("no")
                #mask[i,j] = 0

    #-----------------Select 15 TRIPLETS FROM SEGMENTED IMAGE----------------------------
    points = []
    # CALCULATE THE 15 TRIPLETS COORDINATES THAT HAVE THE MAX DIST WITH dist = numpy.linalg.norm(a-b)
    # Body-part detection 
    for i in range(15):
        points.append(int(random.triangular(0,len(triplets)))) # len(triplets) bc of all possible triplets on segmentated img
    
    from operator import itemgetter
    #-----------------15 TRIPLETS SELECTION-------------------------------------
    sel  = itemgetter(*points)(triplets)
    #print("Selected 15 triplets: ", len(sel))
    #print(sel)

    #----------------SELECT AREAS---------------------------------------------------------
    #print("x ", min(sel))
    triplet_min = min(sel)
    #print("y ", max(sel))
    triplet_max = max(sel)
    x_i,y_i,l_i = triplet_min
    x_f,y_f,l_f = triplet_max

    c1 = x_i,y_i
    c2 = x_f,y_i
    c3 = x_f,y_f
    c4 = x_i,y_f

    rect = c1 + c2 + c3 + c4 + c1 # 5 pairs 
    #print("Polygon coordenates ",rect)

    #plt.imshow(img)
    #plt.colorbar()
    #plt.show()

    final = sel + rect
    #print(final)

    return final

def feature_extraction(path, label): # ADAPT ANNOTATIONS CORRECTLY & POLYGON + AREA EXTRACTION
  #ima = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  
  a = segmentation(path, label)
  
  #contrast = np.amax(ima)-np.amin(ima)
  #sdev = np.std(ima)
  #ssd_blur, blur_count = homogeinty(path)
  #features = np.array([label, mean, contrast, sdev, ssd_blur, blur_count])
  #features = np.array([label, mean])<
  
  return a

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