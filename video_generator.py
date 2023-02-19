import cv2
import themis_imager_readfile
import multiprocessing
import os
import gzip
import logging
import datetime

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


def pgm_images_to_mp4(decompressed_folder_path, video_folder_path):
    logging.info('video convertion start')
    # Initialize a list to store the paths to the pgm files
    pgm_file_paths = []
    for file_name in os.listdir(decompressed_folder_path):
        if not file_name.startswith('.'):
            pgm_file_paths.append(os.path.join(
                decompressed_folder_path, file_name))

    # Sort the list of pgm files -- needed as we decompressed using multithreading
    pgm_file_paths.sort()

    # Initialize the video writer
    video_path = os.path.join(video_folder_path, file_name[:12]+'video.mp4')
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (256, 256))

    # Iterate over the pgm files and write them to the video file
    for image_path in pgm_file_paths:
        # file_name = image_path[-36:]
        # date_text = file_name[:8]
        # frame_text = file_name[9:13]
        frame_number = image_path.split('/')[-1]
        image = cv2.imread(image_path)  # , cv2.IMREAD_GRAYSCALE
        # cv2.putText(
        #     image, date_text+' ' +
        #     frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5, (209, 80, 0, 255), 1)
        cv2.putText(image, frame_number,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
        video_writer.write(image)

    # logging.info(file_name + "//" + date_text + frame_text)
    # Release the video writer
    video_writer.release()
    logging.info('video convertion end')

if __name__ == '__main__':
    child_folder_path = '/Volumes/Garrys_T7/rtroyer-useful-functions/image/rank/tmp/2020-01-04/ut00'
    outer_folder_path = '/Volumes/Garrys_T7/rtroyer-useful-functions/image/rank/tmp/2020-01-04'

    

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
    pgm_images_to_mp4(decompressed_folder_path, outer_folder_path)
