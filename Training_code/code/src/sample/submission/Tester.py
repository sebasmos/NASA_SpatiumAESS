
print('###########################################')
print('########## Test.sh  ############')
print('###########################################')

import numpy as np 
import sys,os
import cv2
import pandas as pd
from utils import *
from pathlib import Path

if __name__ == '__main__':
     # Parse arguments.
     if len(sys.argv) != 3:
         raise Exception('Include the data folder as argument, e.g., ./test.sh ./data/test ./wdata/')
     root_test_data = sys.argv[1]

     root_sol = sys.argv[2]

     # Root to store csv: 
     root_sol_img = os.path.join(root_sol, "images/annotations/")
     root_sol_videos = os.path.join(root_sol, "videos/annotations/")
     try:
         os.makedirs(root_sol_img, exist_ok = True)
         os.makedirs(root_sol_videos, exist_ok =True)
     except: 
         pass

     root_dir = os.path.join(root_test_data) 

     print("Accessed to testing data ", root_dir)
     if 'Training_code' in os.listdir('.'):
        wd = os.path.join(os.getcwd(), "Training_code","code")
        print("working directory is ", wd)
        print(root_dir.split('/'))
        pathfinal=wd
        for x in root_dir.split('/'):
            pathfinal=os.path.join(pathfinal, x)
        root_dir=pathfinal
        print("new working directory is ", root_dir)
     else:
        wd = os.getcwd()
        print("working directory is ", wd)
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
     train_size = len(training_images)
     final = []

     data_train = []

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
         file1 = open(root_sol_img+"solutions.txt","a")
         file1.write(row + " \n")
         file1.close()
         i = i+1
     print("Finished annotation images.. ")
     
     #--------------------------VIDEOS----------------------------------------------------------
     print("Creating wdata/videos/annotations/<id-video>.csv [..]")

     videos = extractVideoNames(video_frames)
     videos_indexes_dict = {vid:1 for vid in videos}
     print(videos)
     topheader = 'Frame #,CLAV,,,RSJC,,,LSJC,,,REJC,,,LEJC,,,RWJC,,,LWJC,,,RHJC,,,LHJC,,,RKJC,,,LKJC,,,RAJC,,,LAJC,,'
     header = ['']+['X','Y','Z']*13
     path = ''
     kk=0
     for frame_path in video_frames:
        
         for v in videos:
             if v in frame_path:
                 print('    {}/{}...'.format(kk+1, len(video_frames)))
                 anotations_path = "{}".format(root_sol_videos)
                 # create anotations folder if not exists
                 try:
                     os.makedirs(anotations_path)
                 except: 
                     pass
                 path_ = "{}{}{}".format(anotations_path,os.sep,v+".csv")            
                 
                 path = frame_path
                 header_str = ",".join(header)+"\n"
                 file1 = open(path_,"a")
                 if videos_indexes_dict[v]==1:
                     print(topheader + " \n" + header_str)
                     file1.write(topheader + " \n" + header_str)
                 triplets, flag = video_feature_extraction(path , 2)
                 id_img = frame_path[len(path) - 15:len(path)]
                 row = str(triplets)[1:-1]
                 print(' {} '.format(row))
                 #record the first column index
                 row = str(videos_indexes_dict[v])+ "," + row
                 row_str = " ".join(map(str, row))
                 file1.write(row + " \n")
                 file1.close()
                 videos_indexes_dict[v]+=1
                 kk = kk+1
     print("Finished annotation videos.. ")