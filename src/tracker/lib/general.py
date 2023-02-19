
# Creates some of the files that are used in the programs and other items
## TODO Create the data files if they are not already done. If possible, have an alternative data file location.

import cv2
import os
import sys
import numpy as np



def create_data_file(file_path):
    # Creates data file with headers.
    with open(file_path, "w") as f:
        f.write("Time, x, y, z\n")

def make_default_folder(data_input, data_output, color):
# Make a new folder to save all of the files into. 
    base_path = os.getcwd()
    dir_path = os.path.abspath(os.path.join(base_path, 'data', data_output , ''))
    data_output_folder = 'default' + color
    data_output_folder_path = os.path.abspath(os.path.join(dir_path , data_output_folder , ''))
    
    # Create a folder to store all of the data in if it does not already exist
    if not os.path.exists(data_output_folder_path):
        os.makedirs(data_output_folder_path)

       
    print('Your folder will be available:\n', data_output_folder_path)
    return data_output_folder, data_output_folder_path

def GUI_creates_an_array_of_csv_files (data_output_folder_path):
# select multiple files is useful when you want to graph many objects
    file_type = '.csv'
    num_files = 1
    dt = np.dtype([ ('filepath', np.unicode_, 60), ('filename', np.unicode_, 30)])    
        
    for file_name in os.listdir(data_output_folder_path):
        if file_name.endswith(file_type):
            print(file_name)
            if num_files == 1:
                file_np = np.array([((data_output_folder_path), (file_name))], dtype=dt)
                file_array = file_np
                print('filenp ', file_np)
            else:
                file_np = np.array([((data_output_folder_path), (file_name))], dtype=dt)
                file_array = np.hstack((file_array,file_np))
                print(file_array)
            num_files +=1
    return file_array

def open_the_video(src):
    video = cv2.VideoCapture(src) # change num based on local machine

    # Exit if video not opened.
    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    # Read first 20 frames of OpenCV video source to warm up
    for _ in range(1, 20):
        ret, _ = video.read()
        if not ret:
            print('Cannot read video file')
            sys.exit()
    return video
