# themis_video_generator
This repository generates videos from THEMIS images for humans to look at and also uses machine learning-based techniques to classify THEMIS images based on aurora types.

1. **video_generator.py** contains the functions that generate videos for THEMIS images from https://data.phys.ucalgary.ca/sort_by_project/THEMIS/. Check out https://github.com/ucalgary-aurora/themis-imager-readfile as well to properly read THEMIS images.
2. **all_tasks.py** generates ML classified txt files for THEMIS images based on *CNN_model/model*. 



> **ucalgary** is the stable branch for these two functions. If you only want the video generation function, use the **video_generate_only** branch. **ucalgary-test** is the unstable branch. We are reconstructing the structure of all_tasks.py so it has lower runtime. 
