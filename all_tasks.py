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
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# get args from command line
if len(sys.argv) > 1:
    args = sys.argv

mp.set_start_method('spawn')
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

    # decompress the images to a dictionary
    # address example: stream0/2011/08/08/mcgr_themis11/ut09/
    # iterate through date folders
    for date_folder_path in subfolder_paths:  # stream0/2011/08/08
        logging.info(f'Processing date_folder_path = {date_folder_path}')

        # Iterate over the child folders (each camera) in the outer folder
        for asi_name in os.listdir(date_folder_path):  # /mcgr_themis11

            # camera_dict example k-v pair: {'atha20200104000206':img[:,:,:]}
            camera_dict = {}

            logging.info(
                f'Processing asi = {asi_name}, date_folder_path = {date_folder_path}')
            asi_folder_path = os.path.join(date_folder_path, asi_name)
            hours = []

            for hour_name in os.listdir(asi_folder_path):  # /ut09
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
                    logging.info(f'skipping date_folder_path = {date_folder_path}, asi = {asi_name}')
                    continue


            except Exception as e:
                logging.CRITICAL(f'Error occurs in making dataframe as {e}')
                sys.exit()

            # try multiprocessing steps
            try: 
                logging.info(f'starting multiprocessing')
                # Create a pool of worker processes
                num_workers = cpu_count()
                pool = Pool(processes=num_workers)
                logging.info(f'pool generated, num_workers = {num_workers}')

                # Map the process_image function to each item in camera_dict using multiprocessing
                results = pool.map(process_image, camera_dict.items())
                logging.info('results generated')

                # Close the pool of worker processes
                pool.close()
                pool.join()
                logging.info('pool joined')    

                # Append the processed rows to the DataFrame
                for result in results:
                    new_row, directory_path, ymd_str = result
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            except Exception as e:
                logging.CRITICAL(f'Error occurs in multiprocessing as {e}')
                sys.exit()

            try:  
                logging.info(f'starting generating output')
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                # needed info: date, time, prediction, prediction_str, confidence
                with open(os.path.join(directory_path, ymd_str+'_'+asi_name+"_classifications.txt"), "w") as f:
                    # create the comment section
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    comment = f"# File created on {now}\n# This file contains the predictions generated by the model.\n\n"
                    f.write(comment)
                    df.to_csv(f, sep='\t', index=False)

                logging.info(f'date_folder_path={date_folder_path}, asi={asi_name} results generated')
            except Exception as e:
                logging.CRITICAL(f'Error occurs in outputing results as {e}')
