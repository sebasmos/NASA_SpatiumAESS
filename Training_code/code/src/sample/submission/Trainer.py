import numpy as np 
import sys,os
import cv2
import pandas as pd
import segmentation from utils



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
        data_train[i, :] = feature_extraction(path, 2)
        i = i+1