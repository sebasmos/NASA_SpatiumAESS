import numpy as np 
import sys,os
import cv2
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils import *
import random

def feature_extraction(path, label): # ADAPT ANNOTATIONS CORRECTLY & POLYGON + AREA EXTRACTION
  ima = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  mean = np.mean(ima)
  #contrast = np.amax(ima)-np.amin(ima)
  #sdev = np.std(ima)
  #ssd_blur, blur_count = homogeinty(path)
  #features = np.array([label, mean, contrast, sdev, ssd_blur, blur_count])
  features = np.array([label, mean])
  return features

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 2:
        raise Exception('Include the data folder as argument, e.g., python Trainer.py training_Data')

    root_dir = sys.argv[1]

    print("Accessed to training data ", root_dir)

    # Load data paths
    training_images = []   # images png + jpg
    img_ann = []        # images annotations
    
    videos = []         # videos
    videos_ann = []     # videos annotations
    video_frames = []   # video frames
    video_frames_ann=[] # video frames annotations

    for root, dirs, files in os.walk(root_dir, topdown=True):
        for name in files:
            path = os.path.join(root, name)            
            if '.png' in path  and 'images' in path:
                training_images.append(path)
                print(path)
            if '.jpg' in path  and 'images' in path:
                training_images.append(path)
            if '.txt' in path and 'images' in path:
                img_ann.append(path)
            if '.mov' in path and 'videos' in path:
                videos.append(path)
            if '.csv' in path and 'videos' in path:
                videos_ann.append(path)
            if '.jpg' in path and 'videos-frames' in path:
                video_frames.append(path)
            #if '.csv' in path and 'videos-frames' in path: #TODO: CHECK ANNOTATIONS
            #    video_frames_ann.append(path)
    
    print('Image training instances: ', len(training_images))
    print('videos training instances: ', len(videos))
    print('video_frames evaluation instances: ', len(video_frames))
    #print(video_frames_ann)

    # Training with images 
    Num_features = 1
    feat_names = ['Polygon']
    train_size = len(training_images)

    data_train = np.zeros((train_size, Num_features+1), dtype=np.float32)
    #data_test = np.zeros((test_size, Num_features+1), dtype=np.float32)
    i = 0
    for path in training_images:
        print('    {}/{}...'.format(i+1, len(training_images)))
        data_train[i, :] = feature_extraction(path, 2)
        i = i+1
    
    # Testing possible outputs
    joints_id = []
    geom = []
    Num_suits = 1 # TO UPDATE ACCORDING TO IMAGE
    Triplets = 15 #
    aux = Num_suits*Triplets
    #joints = np.zeros((train_size, Num_suits*Triplets))

    #train ? 
    id_images = []
    aux = len(path) - 10

    for id, j in enumerate(training_images):

        for i in range(Num_suits):
                #print(" es multiplo de 3")
                x_f,y_f,v_f = gen_array()
                joints_id.append(x_f + y_f + v_f)

                x_geom, y_geom = gen_geometries()
                geom.append(x_geom + y_geom)
                
    print("id_images: ",train_size)
    print("joints_id: ",len(joints_id))
    print("geom: ",len(geom))
    # Store to csv
    df1 = pd.DataFrame(id_images) 
    df2 = pd.DataFrame(joints_id)
    df3 = pd.DataFrame(geom)

    df4 = df1.merge(df2,left_index=True, right_index=True)
    df4.merge(df3,left_index=True, right_index=True)

    df4.to_csv("../solution/training_images.csv")
    print("Finished, stored in: ../solution/training_images.csv ")
    #df4.head(10)