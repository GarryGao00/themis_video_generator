from video_generator import *
from datetime import datetime, timedelta
import sys
import logging

# get args from command line
if len(sys.argv) > 1:
    args = sys.argv

# set the folder path for stream0
stream0_path = '/stream0'

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
def _decompress_pgm_files_to_dict(folder_path, img_dict):
    logging.info('decompress start, hour = '+folder_path[-4:])
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
        dt = datetime.datetime.strptime(strtime, "%Y-%m-%d %H:%M:%S.%f %Z")
        # '20200104000206'
        dt = dt.strftime('%Y%m%d%H%M%S')
        key = meta[frame]['Site unique ID']+dt
        value = img[:,:,frame]
        img_dict[key] = value

    logging.info(folder_path[-4:] + ' decompress done')
    return

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

    except Exception as e:
        print(f'Start or end date not valid, Exception: {e}')
        sys.exit()

    # full_dict example k-v pair: {'stream0/2011/08/08':day_dict}
    full_dict = {}
    # address example: stream0/2011/08/08/mcgr_themis11/ut09/
    # iterate through date folders
    for date_folder_path in subfolder_paths:  # stream0/2011/08/08
        # Iterate over the child folders in the outer folder
        # day_dict example k-v pair: {'atha20200104000206':img[:,:,:]}
        day_dict = {}
        hours = []
        for asi_name in os.listdir(date_folder_path):  # /mcgr_themis11
            asi_folder_path = os.path.join(date_folder_path, asi_name)
            for hour_name in os.listdir(asi_folder_path):  # /ut09
                # check if it is a sub folder
                hour_folder_path = os.path.join(asi_folder_path, hour_name)
                if os.path.isdir(hour_folder_path):
                    hours.append(hour_folder_path)

            for hour in hours:
                _decompress_pgm_files_to_dict(hour, day_dict)
        full_dict[date_folder_path] = day_dict
