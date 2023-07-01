from video_generator import *
from all_tasks_func import *
from datetime import datetime, timedelta
import sys
import logging
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count, get_context
import multiprocessing as mp

# get args from command line
if len(sys.argv) > 1:
    args = sys.argv

# set GPU devices to empty
os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == '__main__':

    # print code start running
    print(f'code running, args = {args[1:]}')

    # init log file
    logging.basicConfig(filename='all_tasks.log',
                        # encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('all_task test code start ' +
                 datetime.now().strftime("%H:%M:%S"))

    # use start_date and end_date to get needed folder paths
    try:
        start_date_str, end_date_str = args[1], args[2]

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        # format: ['stream0/2010/10/01', ...]
        subfolder_paths = get_subfolders_in_range(
            start_date, end_date, folder_path=stream0_path)
        logging.info(
            f'getting paths from {subfolder_paths[0]} to {subfolder_paths[-1]}')
    except Exception as e:
        print(f'Start or end date not valid, Exception: {e}')
        sys.exit()

    # set the num of workers for multiprocessing later. default as the cpu_count.
    try:
        if len(args)>3:
            num_workers = int(args[3])
        else:
            num_workers = cpu_count()
    except Exception as e:
        print(f'Number of processors not valid, Exception: {e}')
        sys.exit()

    # decompress the images to a dictionary
    # address example: stream0/2011/08/08/mcgr_themis11/ut09/
    # iterate through date folders
    for date_folder_path in subfolder_paths:  # stream0/2011/08/08
        logging.info(
            f'Processing date_folder_path = {date_folder_path}, {datetime.now().strftime("%H:%M:%S")}')

        # Iterate over the child folders (each camera) in the outer folder
        for asi_name in os.listdir(date_folder_path):  # /mcgr_themis11
            # camera_dict example k-v pair: {'atha20200104000206':img[:,:,:]}
            camera_dict = {}

            logging.info(
                f'Processing asi = {asi_name}')
            asi_folder_path = os.path.join(date_folder_path, asi_name)
            hours = []

            for hour_name in os.listdir(asi_folder_path):  # /ut09 get hour folders
                # check if it is a sub folder
                hour_folder_path = os.path.join(asi_folder_path, hour_name)
                if os.path.isdir(hour_folder_path):
                    hours.append(hour_folder_path)

            for hour in hours:
                decompress_pgm_files_to_dict(hour, camera_dict)

            logging.info(f'asi_name = {asi_name}, date = {date_folder_path} decompressed')
            try:
                # init a dataframe to store information
                df = pd.DataFrame(
                    columns=['date', 'time', 'prediction', 'prediction_str', 'confidence'])
                
                if not camera_dict:
                    logging.info(
                        f'DATE SKIPPED: camera_dict empty, asi_name = {asi_name}, date = {date_folder_path}')
                    #del camera_dict
                    continue


            except Exception as e:
                logging.CRITICAL(f'Error occurs in making dataframe as {e}')
                logging.CRITICAL(f'DATE SKIPPED: asi_name = {asi_name}, date = {date_folder_path}')
                continue # if exception, go to next asi camera

            # try multiprocessing steps
            try: 
                logging.info(f'starting multiprocessing')
                # Create a pool of worker processes
                num_workers = num_workers
                pool = get_context("spawn").Pool(processes=num_workers)
                logging.info(f'pool generated, num_workers = {num_workers}')

                # Map the process_image function to each item in camera_dict using multiprocessing
                results = pool.map(process_image_clahe, camera_dict.items())
                frames, directory_paths, ymd_strs, time_strs = results
                #del camera_dict
                preds = model(frames)
                prediction_nums = list(map(np.argmax, preds))
                confidences = list(map(np.max, preds))
                prediction_strs = [lb.classes_[item] for item in prediction_nums] 
                new_rows = pd.DataFrame({'date': ymd_strs, 'time': time_strs, 'prediction': prediction_nums, 'prediction_str': prediction_strs, 'confidence': confidences}) 
                # Close the pool of worker processes
                pool.close()
                pool.join()
                logging.info(f'Pool joined')
                # Append the processed rows to the DataFrame
                for frame in frames:
                    frame = frame
                    df = pd.concat([df, new_rows], ignore_index=True)
                
                #del results
                logging.info(f'dataframe generated')
        

            except Exception as e:
                logging.CRITICAL(f'Error occurs in multiprocessing as {e}')
                logging.CRITICAL(
                    f'DATE SKIPPED: asi_name = {asi_name}, date = {date_folder_path}')
                continue  # if exception, go to next asi camera

            try:  
                directory_path = directory_path[-1]
                ymd_str = ymd_strs[-1]
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                # needed info: date, time, prediction, prediction_str, confidence
                logging.info(f'writing dataframe to txt file.')
                with open(os.path.join(directory_path, ymd_str+'_'+asi_name+"_classifications.txt"), "w") as f:
                    # create the comment section
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    comment = f"# File created on {now}\n# This file contains the predictions generated by the model.\n\n"
                    f.write(comment)
                    df.to_csv(f, sep='\t', index=False)
                #del df
                logging.info(f'date_folder_path={date_folder_path}, asi={asi_name} results generated, time = {datetime.now().strftime("%H:%M:%S")}')
            except Exception as e:
                logging.CRITICAL(
                    f'Error occurs in outputing resultsfor asi_name = {asi_name}, date = {date_folder_path} as {e}')
                continue
