import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import core.utils as utils

from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from yolo_pdetector import detect_people

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_boolean('show', True, 'show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    input_size = 416
    iou = 0.45
    score = 0.50
    nms_max_overlap = 1.0
       
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
 
    # Load standard tensorflow saved model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # Paths for videos
    video_dir='./Videos'
    #video_names=['abandonedBox','backdoor','badminton','busStation','copyMachine','cubicle', 'office','overpass', 'fall'] 
    video_names=['badminton'] 

    for video_name in video_names:
        file_frames_path = "{}/{}/dataIn.txt".format(video_dir, video_name)
        list_frames = open(file_frames_path, "r")
        num_lines = len(open(file_frames_path).readlines())
        output_path = "{0}/Results/{1}.idl".format(video_dir, video_name)
        frame_num = 1 
        f = open(output_path, "w").close       
        for image_name in list_frames:            
            if not image_name.strip():
                continue
            print("{}. Frame: {} of {} ".format(video_name, frame_num, num_lines)) 
            path_img = "{}/{}".format(video_dir, image_name[0:-1])
            detect_people(path_img, infer, encoder, tracker, input_size, iou,
             score, nms_max_overlap, FLAGS.show, FLAGS.info, output_path)
            frame_num+=1
             
    print("Done")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    