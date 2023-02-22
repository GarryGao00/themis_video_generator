import cv2
import themis_imager_readfile
import multiprocessing
import os
import logging
import datetime
import numpy
import subprocess

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
def download_themis_images(date, asi, folder_path='./images',force=0):
    logging.info('Downloading {} {}.'.format(asi, date.date()))

    # check and create destination folder if not exists
    date_string = date.strftime('%Y-%m-%d')
    full_path = folder_path+'/'+asi+'/'+date_string
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    # if path exists and not forcing, return path
    elif force == 0:
        logging.info('Already downloaded at {}.'.format(full_path))
        return full_path
    # if path exists and forcing, delete everything in the folder
    elif force == 1:
        for filename in os.listdir(full_path):
            file_path = os.path.join(full_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        # Delete all subdirectories in the directory
        for dirpath, dirnames, filenames in os.walk(full_path, topdown=False):
            for dirname in dirnames:
                os.rmdir(os.path.join(dirpath, dirname))

    # dir_url - destination directory url
    date_string_url = date.strftime('%Y/%m/%d/')
    themis_url = 'data.phys.ucalgary.ca/data/sort_by_project/THEMIS/asi/stream0/'
    dir_url = themis_url + date_string_url + asi + '*/'

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


def list_and_decompress_pgm_files(img_folder_path):
    logging.info('list_and_decompress start')
    tic = datetime.datetime.now()

    # Create 'decompressed' folder
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
        folder_path = os.path.join(img_folder_path, folder_name)
        # check if it is a sub folder
        if os.path.isdir(folder_path):
            hours.append(folder_path)

    for hour in hours:
        _decompress_pgm_files(hour, decompressed_folder_path)

    toc = datetime.datetime.now()
    logging.info('list_and_decompress done in ' + str((toc-tic).total_seconds()) + ' seconds')
    return decompressed_folder_path


def pgm_images_to_mp4(decompressed_folder_path, video_folder_path='./videos', file_suffix='video.mp4', method='None'):
    logging.info('video convertion start')
    tic = datetime.datetime.now()

    # create video_folder if not exists
    if not os.path.exists(video_folder_path):
            os.makedirs(video_folder_path)

    # Initialize a list to store the paths to the pgm files
    pgm_file_paths = []
    for file_name in os.listdir(decompressed_folder_path):
        if not file_name.startswith('.'):
            pgm_file_paths.append(os.path.join(
                decompressed_folder_path, file_name))

    # Sort the list of pgm files -- needed as we decompressed using multithreading
    pgm_file_paths.sort()

    with multiprocessing.Pool() as pool:
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
    video_path = os.path.join(video_folder_path, file_name[:12]+file_suffix)
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (256, 256), 0)

    frame_number = 0
    for image in processed_images:
        cv2.putText(image, str(frame_number), (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
        video_writer.write(image)
        frame_number = frame_number + 1
        
    # # Iterate over the pgm files and write them to the video file
    # for image_path in pgm_file_paths:
    #     frame_number = image_path.split('/')[-1]
    #     if method == 'bytescale':
    #         image = _bytescale(image_path)
    #     elif method == 'eqhist':
    #         image = _eqhist(image_path)
    #     elif method == 'relu':
    #         image = _relu(image_path)
    #     elif method == 'clahe':
    #         clahe = cv2.createCLAHE(clipLimit=30)
    #         image = _clahe(image_path, clahe)
    #     else:
    #         image = cv2.imread(image_path, 0)

    #     cv2.putText(image, frame_number,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
    #     video_writer.write(image)

    # Release the video writer
    video_writer.release()

    toc = datetime.datetime.now()
    logging.info('video convertion end in ' +
                 str(round((toc-tic).total_seconds(), 2))+' seconds')



if __name__ == '__main__':
    # img_folder_path = ''
    # video_folder_path = ''

    logging.basicConfig(filename='video_generator.log',
                        # encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('MiniVideoGenerator test code start ' +
                 datetime.datetime.now().strftime("%H:%M:%S"))

    mydate = datetime.datetime(2016, 10, 13, 0, 0, 0)
    asi = 'gako'
    img_folder_path = download_themis_images(mydate, asi, force=0)
    decompressed_folder_path = list_and_decompress_pgm_files(img_folder_path)
    pgm_images_to_mp4(decompressed_folder_path, file_suffix='clahe1.mp4', method='clahe')
