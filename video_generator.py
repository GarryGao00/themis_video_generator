import cv2
import themis_imager_readfile
import multiprocessing
import os
import logging
import datetime
import numpy
import subprocess
import shutil
import h5py
import re
from scipy.io import readsav

# timing decorator
def _timeit(func):
    def wrapper(*args, **kwargs):
        tic = datetime.datetime.now()
        result = func(*args, **kwargs)
        toc = datetime.datetime.now()
        logging.info(f'{func.__name__} took {toc-tic} seconds')
        return result
    return wrapper

def _decompress_pgm_files(folder_path, decompressed_folder_path):
    logging.info('decompress start, hour = '+folder_path[-4:])
    # folder_path: str, should be ut** folder path

    # get all compressed images absolute path in the folder, exclude hidden files and different shape files
    file_names = os.listdir(folder_path)
    file_names = [folder_path+'/'+f for f in file_names if 'full' in f and not f.startswith('.')]

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
        temp_file_name = meta[frame]['Site unique ID']+dt+'.pgm'
        temp_path = os.path.join(decompressed_folder_path, temp_file_name)
        cv2.imwrite(temp_path, img[:, :, frame])

    logging.info(folder_path[-4:]+ ' decompress done')
    return 

# bytescale function from UCalgary
def _bytescale(image_path, cmin=None, cmax=None, high=65535, low=0):
    image = cv2.imread(image_path, 0)

    if high > 65535:
        raise ValueError("`high` should be less than or equal to 65535.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError(
            "`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = image.min()
    if cmax is None:
        cmax = image.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (image - cmin) * scale + low
    im_scaled =  (bytedata.clip(low, high) + 0.5).astype(numpy.uint16)
    image = (im_scaled // 256)
    image = numpy.uint8(image)
    return image

# equalize histogram
def _eqhist(image_path):
    image = cv2.imread(image_path, 0)
    image = cv2.equalizeHist(image)  # equalize histogram
    image = numpy.uint8(image)
    return image

# contrast limited adaptive histogram equalization
def _clahe(image_path):
    image = cv2.imread(image_path, 0)
    clahe = cv2.createCLAHE(clipLimit=30)
    image = clahe.apply(image)
    return image

def _relu(image_path, low = 0, pivot=0.02, ratio=1.7):
    # process the data in relu
    def _relu_help(data, pivot=pivot, low=low, ratio=ratio):
        return numpy.maximum(data+low, ratio*(data-pivot)+low)

    image = cv2.imread(image_path, 0)
    image = _relu_help(image)
    image = image.clip(low, 255).astype(numpy.uint8)

    return image

# read the image without edit
def _read_img(image_path):
    image = cv2.imread(image_path, 0)
    return image

# download images from UCalgary. 
# Date: datetime object; force: 0-check if downloaded first, 1-delete existing date and redownload
# Return: fullpath, a string to the folder of the downloaded images
@_timeit
def download_themis_images(date, asi, folder_path='./images',force=0, skymap=1):
    logging.info(f'Downloading {asi} {date.date()}, force = {force}, skymap = {skymap}')

    # check and create destination folder if not exists
    date_string = date.strftime('%Y-%m-%d')
    full_path = folder_path+'/'+asi+'/'+date_string
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    # if path exists and not forcing, return path
    elif force == 0:
        logging.info(f'Already downloaded at {full_path}.')
        return full_path
    # if path exists and forcing, delete everything in the folder
    elif force == 1:
        try: 
            shutil.rmtree(full_path)
            os.makedirs(full_path)
        except Exception as e:
            logging.critical(f'folder cannot be removed at {full_path}')
            logging.critical('Exception: {}'.format(e))

    # dir_url - destination directory url
    date_string_url = date.strftime('%Y/%m/%d/')
    themis_url = 'data.phys.ucalgary.ca/data/sort_by_project/THEMIS/asi/stream0/'
    dir_url = themis_url + date_string_url + asi + '*/'

    # Download skymap
    if skymap:
        logging.info('Downloading skymap')
        skymap_url = 'data.phys.ucalgary.ca/data/sort_by_project/THEMIS/asi/skymaps/' + asi + '/'

        # find skymap dirs online
        try:
            skymap_dirs = subprocess.check_output(['rsync',
                                                'rsync://' + skymap_url]).splitlines()
            skymap_dirs = [str(d.split(b' ')[-1], 'UTF-8')
                        for d in skymap_dirs[1:]]
        except Exception as e:
            logging.critical('Unable to access skymap server: {}. '
                            'Server may be down. Stopping.'.format(skymap_url))
            logging.critical('Exception: {}'.format(e))
            raise

        # Convert to datetimes
        skymap_dates = [d.split('_')[1] for d in skymap_dirs]
        skymap_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in skymap_dates]
        time_diffs = numpy.array([(date - d).total_seconds() for d in skymap_dates])
        skymap_dir = skymap_dirs[numpy.where(time_diffs > 0, time_diffs, numpy.inf).argmin()]

        skymap_url = skymap_url + skymap_dir + '/'
        try:
            subprocess.run(['rsync', '-vzrt', 'rsync://' + skymap_url + '*.sav',
                            full_path], stdout=subprocess.DEVNULL)
            logging.info('Successfully downloaded skymap.'
                        ' It is saved at {}.'.format(full_path))
        except Exception as e:
            logging.critical(
                'Unable to download skymap:{}. Stopping.'.format(skymap_url))
            logging.critical('Exception: {}'.format(e))
            raise

    # Download images
    logging.info('Downloading images from {}...'.format(dir_url))
    try:
        subprocess.run(['rsync', '-vzrt', '--size-only', '--ignore-existing', 'rsync://' + dir_url,
                        full_path], stdout=subprocess.DEVNULL)
        logging.info('Successfully downloaded at {}.'.format(folder_path))
    except Exception as e:
        logging.critical('Unable to download images:{}.'.format(dir_url))
        logging.critical('Exception: {}'.format(e))
        raise

    return full_path

@_timeit
def list_and_decompress_pgm_files(img_folder_path):
    logging.info(f'list_and_decompress start for {img_folder_path}')

    # Check and Create 'decompressed' folder
    parent_folder_path = os.path.dirname(img_folder_path)
    current_folder_name = os.path.basename(img_folder_path)
    decompressed_folder_path = os.path.join(parent_folder_path, current_folder_name+'-decompressed')
    if os.path.isdir(decompressed_folder_path):
        logging.info('Already decompressed at '+ decompressed_folder_path)
        return decompressed_folder_path

    os.makedirs(decompressed_folder_path, exist_ok=True)
    logging.info('Decompressed folder created')

    # store the paths of the hour folder
    hours = []

    # Iterate over the child folders in the outer folder
    for folder_name in os.listdir(img_folder_path):
        # if it is the skymap, move it to the decompressed folder
        if folder_name.endswith('.sav') and not folder_name.startswith('.'):
            skymap_path = os.path.join(img_folder_path, folder_name)
            shutil.copy(skymap_path, decompressed_folder_path)
            continue
        # check if it is a sub folder
        folder_path = os.path.join(img_folder_path, folder_name)
        if os.path.isdir(folder_path):
            hours.append(folder_path)

    for hour in hours:
        _decompress_pgm_files(hour, decompressed_folder_path)

    toc = datetime.datetime.now()
    logging.info('list_and_decompress done')
    return decompressed_folder_path

@_timeit
def pgm_images_to_mp4(decompressed_folder_path, video_folder_path='./videos', file_suffix='video.mp4', method='None', processes=8):
    logging.info('video convertion start')

    # create video_folder if not exists
    if not os.path.exists(video_folder_path):
            os.makedirs(video_folder_path)

    # Initialize a list to store the paths to the decompressed pgm files
    pgm_file_paths = []
    for file_name in os.listdir(decompressed_folder_path):
        if not file_name.startswith('.') and file_name.endswith('.pgm'):
            pgm_file_paths.append(os.path.join(
                decompressed_folder_path, file_name))

    # Sort the list of pgm files -- needed as we decompressed using multithreading
    pgm_file_paths.sort()

    with multiprocessing.Pool(processes=processes) as pool:
        # process all images using the pool of worker processes
        if method == 'bytescale':
            processed_images = pool.map(_bytescale, pgm_file_paths)
        elif method == 'eqhist':
            processed_images = pool.map(_eqhist, pgm_file_paths)
        elif method == 'relu':
            processed_images = pool.map(_relu, pgm_file_paths)
        elif method == 'clahe':
            processed_images = pool.map(_clahe, pgm_file_paths)
        else:
            processed_images = pool.map(_read_img, pgm_file_paths)

    # Initialize the video writer
    camera_date = pgm_file_paths[0].split('/')[-1][:12]
    video_path = os.path.join(
        video_folder_path, camera_date+file_suffix)
    logging.info(f'video_path = {video_path}, file name = {camera_date}')
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (256, 256), 0)

    frame_number = 0
    for image in processed_images:
        cv2.putText(image, str(frame_number), (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
        video_writer.write(image)
        frame_number = frame_number + 1

    # Release the video writer
    video_writer.release()
    logging.info(f'video converted at {video_path}')
    return

@_timeit
def pgm_images_to_h5(decompressed_folder_path, h5_folder_path='./h5s', file_suffix='.h5', method='None', processes=8):
    logging.info('h5 convertion start')

    # create h5 file folder if not exists
    if not os.path.exists(h5_folder_path):
        os.makedirs(h5_folder_path)

    # Initialize a list to store the paths to the decompressed pgm files and read in skymap
    pgm_file_paths = []
    timestamps = []
    for file_name in os.listdir(decompressed_folder_path):
        if not file_name.startswith('.'):
            if file_name.endswith('.pgm'):
                # extract the time
                timestamp_str = re.findall(r'\d{14}', file_name)[0]
                timestamp = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                timestamps.append(timestamp)
                # add pgm image path
                pgm_file_paths.append(os.path.join(decompressed_folder_path, file_name))
            elif file_name.endswith('.sav'):
                skymap_path = os.path.join(decompressed_folder_path, file_name)
    
    # Sort the list of pgm files -- needed as we decompressed using multithreading
    pgm_file_paths.sort()

    # process the images
    with multiprocessing.Pool(processes=processes) as pool:
        # process all images using the pool of worker processes
        if method == 'bytescale':
            processed_images = pool.map(_bytescale, pgm_file_paths)
        elif method == 'eqhist':
            processed_images = pool.map(_eqhist, pgm_file_paths)
        elif method == 'relu':
            processed_images = pool.map(_relu, pgm_file_paths)
        elif method == 'clahe':
            processed_images = pool.map(_clahe, pgm_file_paths)
        else:
            processed_images = pool.map(_read_img, pgm_file_paths)
    
    # Stack the images into a 3D NumPy array
    processed_images_array = numpy.stack(processed_images, axis=-1)
    #logging.info(f'processed_images shape = {processed_images_array.shape}')
    # Stack the timestamps into the format
    timestamps_array = numpy.array([int(t.timestamp()) for t in timestamps])

    try:
            # Try reading IDL save file
        skymap = readsav(skymap_path, python_dict=True)['skymap']

        # Get arrays
        skymap_alt = skymap['FULL_MAP_ALTITUDE'][0]
        skymap_glat = skymap['FULL_MAP_LATITUDE'][0][:, 0:-1, 0:-1]
        skymap_glon = skymap['FULL_MAP_LONGITUDE'][0][:, 0:-1, 0:-1]
        skymap_elev = skymap['FULL_ELEVATION'][0]
        skymap_azim = skymap['FULL_AZIMUTH'][0]

        logging.info('Read in skymap file from: {}'.format(skymap_path))

    except Exception as e:
        logging.error('Unable to read skymap file, creating file without it.')
        logging.error('Exception: {}'.format(e))

        skymap_alt = numpy.array(['Unavailable'])
        skymap_glat = numpy.array(['Unavailable'])
        skymap_glon = numpy.array(['Unavailable'])
        skymap_elev = numpy.array(['Unavailable'])
        skymap_azim = numpy.array(['Unavailable'])
    
    # Initialize the h5 file 
    camera_date = pgm_file_paths[0].split('/')[-1][:12]
    h5_path = os.path.join(h5_folder_path, camera_date+file_suffix)
    logging.info(f'video_path = {h5_path}, file name = {camera_date}')

    # Write in information
    with h5py.File(h5_path, 'w') as h5f:

        # Initialize the datasets for images and timestamps
        img_ds = h5f.create_dataset('images', dtype='uint8', 
                                    data=processed_images_array)

        time_ds = h5f.create_dataset('timestamps', dtype='uint64',
                                     data = timestamps_array)

        alt_ds = h5f.create_dataset('skymap_alt', shape=skymap_alt.shape,
                                    dtype='float', data=skymap_alt)

        glat_ds = h5f.create_dataset('skymap_glat', shape=skymap_glat.shape,
                                     dtype='float', data=skymap_glat)

        glon_ds = h5f.create_dataset('skymap_glon', shape=skymap_glon.shape,
                                     dtype='float', data=skymap_glon)

        elev_ds = h5f.create_dataset('skymap_elev', shape=skymap_elev.shape,
                                     dtype='float', data=skymap_elev)

        azim_ds = h5f.create_dataset('skymap_azim', shape=skymap_azim.shape,
                                     dtype='float', data=skymap_azim)
        
        # Add attributes to datasets
        time_ds.attrs['about'] = ('UT POSIX Timestamp.'
                                  ' Use datetime.fromtimestamp '
                                  'to convert. Time is start of image.'
                                  ' 1 second exposure.')
        img_ds.attrs['wavelength'] = 'white'
        # img_ds.attrs['station_latitude'] = float(meta[0]['Geodetic latitude'])
        # img_ds.attrs['station_longitude'] = float(meta[0]['Geodetic Longitude'])
        alt_ds.attrs['about'] = 'Altitudes for different skymaps.'
        glat_ds.attrs['about'] = 'Geographic latitude at pixel corner, excluding last.'
        glon_ds.attrs['about'] = 'Geographic longitude at pixel corner, excluding last.'
        elev_ds.attrs['about'] = 'Elevation angle of pixel center.'
        azim_ds.attrs['about'] = 'Azimuthal angle of pixel center.'
    
    logging.info(f'h5 file converted at {h5_path}')
    return

