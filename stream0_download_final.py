from video_generator import *
import datetime as dt
import pandas as pd
import os

if __name__ == '__main__':
    # img_folder_path = ''
    # video_folder_path = '/Volumes/Garrys_T7/themis_video_generator/videos'

    logging.basicConfig(filename='video_generator.log',
                        # encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('MiniVideoGenerator test code start ' +
                 datetime.datetime.now().strftime("%H:%M:%S"))

    # import files
    objects = 'classification-03022023.xlsx'

    # preview files
    file = pd.read_excel(objects, header=0, skiprows=1)
    file.columns = file.columns.str.lower()

    # take out dates and camera name
    dcs = []  # set for camera-datetime object pairs
    ds = [] # set for datetimes

    camerass = file.loc[:, "camera"]
    years = file.loc[:, "year"]
    months = file.loc[:, "month"]
    days = file.loc[:, "date"]

    for i in range(len(years)):
        cameras = camerass[i].split(',')
        year = years[i]
        month = months[i]
        day = days[i]
        date = f'{year}-{month}-{day}'
        ds.append(dt.datetime.strptime(date, '%Y-%m-%d'))

        for camera in cameras:
            tempdcs = (camera.strip(), dt.datetime.strptime(date, '%Y-%m-%d'))
            dcs.append(tempdcs)
            

    dcs=list(set(dcs))
    ds=list(set(ds))

    # print(len(ds))
    # print(len(dcs))
    # print(dcs[:5])

    themis_url = 'data.phys.ucalgary.ca/data/sort_by_project/THEMIS/asi/stream0/'
    folder_path = 'stream0'

    for camera_datetime in dcs:
        station, mydate = camera_datetime
        date_string_url = mydate.strftime('%Y/%m/%d/')    
        dir_url = os.path.join(themis_url, date_string_url, station+'*/')
        full_path = os.path.join(folder_path, date_string_url, station+'_themis')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        try:
            subprocess.run(['rsync', '-vzrt', '--size-only', '--ignore-existing', 'rsync://' + dir_url,
                            full_path], stdout=subprocess.DEVNULL)
            logging.info(
                f'Successfully downloaded at {mydate} at {folder_path}')
        except Exception as e:
            logging.info(f'error processing {mydate} {station}, error={e}')

        break
