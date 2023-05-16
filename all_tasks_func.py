from video_generator import *
from datetime import datetime, timedelta
import logging
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import os

# set the folder path for stream0
stream0_path = '/home/garry/pa_project/stream0'
# stream0_path = 'D:\stream0'

# load trained model
model_path = '/home/garry/pa_project/themis_video_generator/CNN_model'
# model_path = 'F:\pa_sample_models\CNN model'
model = load_model(
    os.path.join(model_path, 'model', 'CNN_0119.model'))

# load the binarized class labels
lb_path = os.path.join(model_path, "model/lb_3c.pickle")
lb = pickle.loads(open(lb_path, "rb").read())

# Predictions queue. The prediction is smoothed by
# the averarge of past "maxlen" frames
Q = deque(maxlen=20)

# np.array to cut the bourndary of the frames
elev_angle = np.load(os.path.join(model_path, "T_angle.npy"))
angle = 15

# get the dates available between start_date and end_date in folder_path that points to stream0 folder
def get_subfolders_in_range(start_date, end_date, folder_path=stream0_path):
    subfolder_paths = []
    current_date = start_date
    while current_date <= end_date:
        year = str(current_date.year)
        month = f"{current_date.month:02d}"
        day = f"{current_date.day:02d}"
        subfolder_path = os.path.join(folder_path, year, month, day)
        if os.path.exists(subfolder_path):
            subfolder_paths.append(subfolder_path)
        current_date += timedelta(days=1)
    return subfolder_paths

# helper function that decompress one folder
def decompress_pgm_files_to_dict(folder_path, img_dict):
    logging.info('decompressing hour = '+folder_path[-4:]+'  '+folder_path)
    # folder_path: str, should be ut** folder path

    # get all compressed images absolute path in the folder, exclude hidden files and different shape files
    file_names = os.listdir(folder_path)
    file_names = [folder_path+'/' +
                  f for f in file_names if 'full' in f and not f.startswith('.')]

    # read the images using themis_imager_readfile - input is the list of absolute paths to compressed images
    img, meta, problematic_files = themis_imager_readfile.read(file_names)
    frame_num = img.shape[2]

    for frame in range(frame_num):
        # '2020-01-04 00:02:06.053611 UTC'
        strtime = meta[frame]['Image request start']
        # 'datetime.datetime(2020, 1, 4, 0, 2, 6, 53611)'
        dt = datetime.strptime(strtime, "%Y-%m-%d %H:%M:%S.%f %Z")
        # '20200104000206'
        dt = dt.strftime('%Y%m%d%H%M%S')
        key = meta[frame]['Site unique ID']+dt
        value = img[:, :, frame]
        img_dict[key] = value

    return

# input should be a frame/image, output (predition, label)
def pred_frame(image):
    try:
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # convert the frame to RGB color
        frame = cv2.resize(frame, (256, 256)).astype("float32") # resize the frame to 256 by 256 to cut the boundary
        frame[elev_angle < angle] = 0 #cut the boundary
        frame = cv2.resize(frame, (224, 224)).astype("float32") # resize the frame to 224 by 224 for prediction
        preds = model.predict(np.expand_dims(frame, axis=0))[0] # prediction
        i = np.argmax(preds)
        confidence = np.max(preds)
        label = lb.classes_[i]
        return preds, label, i, confidence
    except Exception as e:
        logging.CRITICAL(f'unable to pred_frame, error = {e}')
        return e

    


def process_image(item):
    key, value = item
    dt = datetime.strptime(key[4:], '%Y%m%d%H%M%S')
    year, month, day = str(dt.year), str(dt.month), str(dt.day)
    directory_path = os.path.join(year, month, day)
    ymd_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H:%M:%S')

    # process the image using clahe
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    value = cv2.convertScaleAbs(clahe.apply(value), alpha=(255.0/65535.0))

    # classification name format: YYYY/MM/DD/YYYYMMDD_site_themis##_classifications.txt
    preds, prediction_str, prediction, confidence = pred_frame(value)
    new_row = {'date': ymd_str, 'time': time_str, 'prediction': prediction,
               'prediction_str': prediction_str, 'confidence': confidence}
    
    return new_row, directory_path, ymd_str
