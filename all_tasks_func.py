from video_generator import *
from datetime import datetime, timedelta
import sys
import logging

# set the folder path for stream0
stream0_path = 'D:\stream0'

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
