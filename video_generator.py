import cv2
import themis_imager_readfile
import multiprocessing
import os
import gzip
import logging
import datetime
import numpy

def _decompress_pgm_files(folder_path, decompressed_folder_path):
    logging.info('decompress start, hour = '+folder_path[-4:])
    '''
    folder_path: str, should be ut** folder path
    '''

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
    
    # logging.info('listdir = '+str(os.listdir(folder_path)))

    logging.info(folder_path[-4:]+ ' decompress done')
    return 


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

def _eqhist(image_path):
    image = cv2.imread(image_path, 0)
    image = cv2.equalizeHist(image)  # equalize histogram
    image = numpy.uint8(image)
    return image

def _relu(image_path, low = 0, pivot=0.02, ratio=1.7):
    image = cv2.imread(image_path, 0)

    # process the data in relu
    def _relu_help(data, pivot=pivot, low=low, ratio=ratio):
        return numpy.maximum(data+low, ratio*(data-pivot)+low)

    image = _relu_help(image)
    image = image.clip(low, 255).astype(numpy.uint8)

    return image


def list_and_decompress_pgm_files(outer_folder_path):
    logging.info('list_and_decompress start')

    # Create 'decompressed' folder
    decompressed_folder_path = os.path.join(outer_folder_path, 'decompressed')
    if os.path.isdir(decompressed_folder_path):
        logging.info('already decompressed at '+ decompressed_folder_path)
        return decompressed_folder_path

    os.makedirs(decompressed_folder_path, exist_ok=True)
    logging.info('decompressed folder created')

    # store the paths of the hour folder
    hours = []

    # Iterate over the child folders in the outer folder
    for folder_name in os.listdir(outer_folder_path):
        folder_path = os.path.join(outer_folder_path, folder_name)
        # check if it is a sub folder
        if os.path.isdir(folder_path):
            hours.append(folder_path)

    for hour in hours:
        _decompress_pgm_files(hour, decompressed_folder_path)

    logging.info('list_and_decompress done')
    return decompressed_folder_path


def pgm_images_to_mp4(decompressed_folder_path, video_folder_path, file_suffix='video.mp4', method='None'):
    logging.info('video convertion start')
    # Initialize a list to store the paths to the pgm files
    pgm_file_paths = []
    for file_name in os.listdir(decompressed_folder_path):
        if not file_name.startswith('.'):
            pgm_file_paths.append(os.path.join(
                decompressed_folder_path, file_name))

    # Initialize the video writer
    video_path = os.path.join(video_folder_path, file_name[:12]+file_suffix)
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (256, 256), 0)

    # Sort the list of pgm files -- needed as we decompressed using multithreading
    pgm_file_paths.sort()
        
    # Iterate over the pgm files and write them to the video file
    for image_path in pgm_file_paths:
        frame_number = image_path.split('/')[-1]
        if method == 'bytescale':
            image = _bytescale(image_path)
        if method == 'eqhist':
            image = _eqhist(image_path)
        if method == 'relu':
            image = _relu(image_path)
        cv2.putText(image, frame_number,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
        video_writer.write(image)

    # logging.info(file_name + "//" + date_text + frame_text)
    # Release the video writer
    video_writer.release()
    logging.info('video convertion end')

if __name__ == '__main__':
    outer_folder_path = '/Volumes/Garrys_T7/rtroyer-useful-functions/image/rank/tmp/2020-01-04'
    video_folder_path = '/Volumes/Garrys_T7/video_folder'

    logging.basicConfig(filename='video_generator.log',
                        # encoding='utf-8',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('MiniVideoGenerator test code start ' +
                 datetime.datetime.now().strftime("%H:%M:%S"))

    decompressed_folder_path = '/Volumes/Garrys_T7/themis_video_generator/decompressed'

    # test_img = '/Volumes/Garrys_T7/rtroyer-useful-functions/image/rank/tmp/2020-01-04/ut00/20200104_0002_rank_themis12_full.pgm.gz'
    # img, meta, problematic_files = themis_imager_readfile.read(test_img)
    # print("Number of images: %d" % (img.shape[2]))

    # decompress_pgm_files tested - working 
    # decompress_pgm_files(child_folder_path, decompressed_folder_path)

    decompressed_folder_path = list_and_decompress_pgm_files(outer_folder_path)

    # decompressed_folder_path = list_and_decompress_pgm_files(outer_folder_path)
    pgm_images_to_mp4(decompressed_folder_path, video_folder_path, file_suffix='bytescale.mp4', method='bytescale')
