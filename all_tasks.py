from video_generator import *
from all_tasks_func import *
from datetime import datetime, timedelta
import sys
import logging

# get args from command line
if len(sys.argv) > 1:
    args = sys.argv


if __name__ == '__main__':

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
        subfolder_paths = get_subfolders_in_range(start_date, end_date, folder_path=stream0_path)
        logging.info(f'getting paths from {subfolder_paths[0]} to {subfolder_paths[-1]}')

    except Exception as e:
        print(f'Start or end date not valid, Exception: {e}')
        sys.exit()

    # full_dict example k-v pair: {'stream0/2011/08/08':day_dict}
    full_dict = {}
    # address example: stream0/2011/08/08/mcgr_themis11/ut09/
    # iterate through date folders
    for date_folder_path in subfolder_paths:  # stream0/2011/08/08
        logging.info(f'Processing date_folder_path = {date_folder_path}')
        # Iterate over the child folders in the outer folder
        # day_dict example k-v pair: {'atha20200104000206':img[:,:,:]}
        day_dict = {}
        for asi_name in os.listdir(date_folder_path):  # /mcgr_themis11
            hours = []
            logging.info(f'Processing asi = {asi_name}, date_folder_path = {date_folder_path}')
            asi_folder_path = os.path.join(date_folder_path, asi_name)
            for hour_name in os.listdir(asi_folder_path):  # /ut09
                # check if it is a sub folder
                hour_folder_path = os.path.join(asi_folder_path, hour_name)
                if os.path.isdir(hour_folder_path):
                    hours.append(hour_folder_path)

            for hour in hours:
                decompress_pgm_files_to_dict(hour, day_dict)
        full_dict[date_folder_path] = day_dict

    print(full_dict.keys())
