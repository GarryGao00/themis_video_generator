#from video_generator import *
#from all_tasks_func_testing import *
from datetime import datetime, timedelta
import sys
import logging
import themis_imager_readfile
#from tensorflow.keras.models import load_model
#from collections import deque
#import numpy as np
import os
from multiprocessing import cpu_count
import psutil
import time
#import pandas as pd
#from multiprocessing import Pool, cpu_count, get_context
#import multiprocessing as mp
#import gc

# get args from command line
if len(sys.argv) > 1:
    args = sys.argv

# set GPU devices to empty
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def ram_used():
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
 
    # Memory usage
    logging.info(f'RAM memory percent  used: {round((used_memory/total_memory) * 100, 2)}')

if __name__ == '__main__':

    # print code start running
    print(f'code running, args = {args[1:]}')

    # init log file
    logging.basicConfig(filename='all_tasks_test.log',
                        # encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('all_task test code start ' +
                 datetime.now().strftime("%H:%M:%S"))

    # use start_date and end_date to get needed folder paths
#    try:
#        start_date_str, end_date_str = args[1], args[2]
#
#        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
#        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
#
#        # format: ['stream0/2010/10/01', ...]
#        subfolder_paths = get_subfolders_in_range(
#            start_date, end_date, folder_path=stream0_path)
#        logging.info(
#            f'getting paths from {subfolder_paths[0]} to {subfolder_paths[-1]}')
#
#    except Exception as e:
#        print(f'Start or end date not valid, Exception: {e}')
#        sys.exit()

#    # set the num of workers for multiprocessing later. default as the cpu_count.
    try:
        if len(args)>3:
            num_workers = int(args[3])
        else:
            num_workers = cpu_count()
    except Exception as e:
        print(f'Number of processors not valid, Exception: {e}')
        sys.exit()
    logging.info('Start of program')
    ram_used()
    # decompress the images to a dictionary
    # address example: stream0/2011/08/08/mcgr_themis11/ut09/
    # iterate through date folders
#    for date_folder_path in os.listdir('stream0_test'):  # stream0/2011/08/08
#        logging.info(
#            f'Processing date_folder_path = {date_folder_path}, {datetime.now().strftime("%H:%M:%S")}')
#
#        # Iterate over the child folders (each camera) in the outer folder
#        for asi_name in os.listdir('stream0_test'):  # /mcgr_themis11
#            ram_used()
#            # camera_dict example k-v pair: {'atha20200104000206':img[:,:,:]}
#            camera_dict = {}#
#
#            logging.info(
#                f'Processing asi = {asi_name}')
#            asi_folder_path = os.path.join(date_folder_path, asi_name)
#            hours = []
#
#            for hour_name in os.listdir('stream0_test'):  # /ut09
#                # check if it is a sub folder
#                hour_folder_path = os.path.join(asi_folder_path, hour_name)
#                if os.path.isdir(hour_folder_path):
#                    hours.append(hour_folder_path)
    #camera_dict = {}
    print('RAM Used (MB):', psutil.virtual_memory()[3]/1000000)
    counter = 0
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for hour in sorted(os.listdir('stream0_test2')):
            logging.info(f'start of hour: {hour}')
            ram_used()
            logging.info(f'Getting images for {hour}')
            folder_path = 'stream0_test2/' + hour
            file_names = os.listdir(folder_path)
            file_names = [folder_path+'/' +
                  f for f in file_names if 'full' in f and not f.startswith('.')]
            
            counter += len(file_names)

            img, meta, problematic_files = themis_imager_readfile.read(file_names, workers=num_workers)
            frame_num = img.shape[2]
            #if frame_num < 10:
               # logging.info('Detected a zero frame file.')
               # camera_dict.clear()
               # continue 

            logging.info(f'Read in images. Frame number: {frame_num}. Problems: {len(problematic_files)}')
            ram_used()
        print(f"{counter} images read")
        print('RAM Used (MB):', psutil.virtual_memory()[3]/1000000)


    del frame_num, counter, img, meta, problematic_files, file_names
    time.sleep(10)
    print(f"after deleting and 10 secs")
    print('RAM Used (MB):', psutil.virtual_memory()[3]/1000000)
            #logging.info(f'Writing frames to dictionary')
           # for frame in range(frame_num):
        #img_dict[all_times[frame]] = all_images[:, :, frame]
        # '2020-01-04 00:02:06.053611 UTC'
               # strtime = meta[frame]['Image request start']
        # 'datetime.datetime(2020, 1, 4, 0, 2, 6, 53611)'
               # dt = datetime.strptime(strtime, "%Y-%m-%d %H:%M:%S.%f %Z")
        # '20200104000206'
               # dt = dt.strftime('%Y%m%d%H%M%S')
               # key = meta[frame]['Site unique ID']+dt
               # value = img[:, :, frame] 

                #camera_dict[key] = value
            #logging.info(f'Finished writing frames to dictionary')
            #ram_used()
            #logging.info('Clearing dictionary')
            #camera_dict.clear()
            #ram_used()
#            logging.info(f'Length of camera dict is: {len(camera_dict.keys())}')
#            del camera_dict
#            gc.collect()
#            ram_used()
                
#            logging.info(f'asi_name = {asi_name}, date = {date_folder_path} decompressed')
#            ram_used()
#            try:
#                # init a dataframe to store information
#                df = pd.DataFrame(
#                    columns=['date', 'time', 'prediction', 'prediction_str', 'confidence'])
#                
#                if not camera_dict:
#                    logging.info(
#                        f'DATE SKIPPED: camera_dict empty, asi_name = {asi_name}, date = {date_folder_path}')
#                    #del camera_dict
#                    continue
#
#
#            except Exception as e:
#                logging.CRITICAL(f'Error occurs in making dataframe as {e}')
#                logging.CRITICAL(f'DATE SKIPPED: asi_name = {asi_name}, date = {date_folder_path}')
#                continue # if exception, go to next asi camera
#
#            ram_used()
#            del camera_dict
#            del df
#            gc.collect()
#            ram_used()
