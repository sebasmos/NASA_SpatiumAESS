import sys,os

import numpy as np 
import sys,os
import cv2
import pandas as pd
from utils import *

if __name__ == '__main__':

     # Parse arguments.
    if len(sys.argv) != 2:
        raise Exception('Include the data folder as argument, e.g., ./train.sh ./data/train')
    root_training_data = sys.argv[1]

    root_dir = os.path.join(root_training_data) 

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
            print(name)
            
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
            if '.csv' in path and 'videos-frames' in path: #TODO: CHECK ANNOTATIONS
                video_frames_ann.append(path)
    print('Image training instances: ', len(training_images))
    print('videos training instances: ', len(videos))
    print('video_frames evaluation instances: ', len(video_frames))
    print(video_frames_ann)
     

     #---------------------FEATURE EXTRACTION--------------------------------------------
    i = 0
     # Training with images 
    Num_features = 1
    feat_names = ['Polygon'] # TODO: EXTRACT TRIPLETS HERE!
    train_size = len(training_images)
    final = []

    data_train = []# np.zeros((train_size, Num_features+1), dtype=np.float32)
    #data_test = np.zeros((test_size, Num_features+1), dtype=np.float32)

    #--------------------------IMAGES-----------------------------------------------------------
    print("Creating wdata/images/annotations/solutions.csv...")
    for path in training_images:
        print('    {}/{}...'.format(i+1, len(training_images)))
       
        triplets, coordinates,flag = feature_extraction(path, 2)
       
        id_img = path[len(path) - 10:len(path)]
       
        if flag==True:
            row = id_img
        else:
            row = id_img + ',' +str(triplets)[1:-1] + ','+ '[' + str(coordinates) + ']' # working with lists
        print(' {} '.format(row))
    
        #file1 = open("solutions_final.csv","a")
        file1 = open("./wdata/images/annotations/solutions_final.csv","a")
        file1.write(row + " \n")
        file1.close()
        i = i+1
print("Storing model to model/...")
