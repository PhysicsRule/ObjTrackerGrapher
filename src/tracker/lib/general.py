
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

       
    print('\nYour folder will be available:\n', data_output_folder_path)
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

def save_video_file(image_file_path, video_img_array, type_of_image, size):
    # Saves the images in a video
    video_file_path = os.path.abspath(os.path.join(image_file_path + type_of_image))  
    out = cv2.VideoWriter(video_file_path +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for i in range(len(video_img_array)):
        out.write(video_img_array[i])
    out.release()

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

def find_objects_to_graph (data_output_folder_path):
# Use the npy file to located the objects or graph each csv file as a different object
# TODO if using the csv file, allow for user to enter the mass of each object
    dt_object = np.dtype([ ('filepath', np.unicode_, 60), ('filename', np.unicode_, 30), ('mass', np.float32)])
    dt_2 = np.dtype([ ('blank1', np.unicode_, 5), ('blank2', np.unicode_, 5), ('filename', np.unicode_, 30), ('blank3', np.unicode_, 5), ('mass', np.float32)])

    npy_found = False
    for np_file_name in os.listdir(data_output_folder_path):
        if np_file_name.endswith('.npy'):
            try:
                np_path = os.path.abspath(os.path.join(data_output_folder_path, np_file_name,''))
                graph_color_ranges = np.load(np_path)
            except:
                print("Couldn't load the npy file with the mass")
            npy_found = True
    if npy_found == False:                
        # TODO have the user enter the data in the future
        print('Since a npy file of the objects was not saved, mass was set to 1.')
        mass = 1
        object_count = 1
        csv_found = False
        for csv_file_name in os.listdir(data_output_folder_path):
            if csv_file_name.endswith('.csv'):
                csv_file_name = os.path.splitext(csv_file_name)[0]
                csv_found = True
                if object_count == 1:
                    graph_color_file = np.array([(('_'),('_'), (csv_file_name),('_'),(mass))], dtype=dt_2)
                    graph_color_ranges = graph_color_file           
                else:
                    graph_color_file = np.array([(('_'),('_'), (csv_file_name),('_'),(mass))], dtype=dt_2)
                    graph_color_ranges = np.hstack((graph_color_ranges,graph_color_file))
                object_count += 1

        if csv_found == False:
            print('*csv Data file was not found.')
        # else:
            # print ('graph_color_ranges', graph_color_ranges)
    object_count = 1
    for _,_,name,_,mass in graph_color_ranges:
        if object_count == 1:
            file_np = np.array([((data_output_folder_path), (name),(mass))], dtype=dt_object)
            csv_files_array = file_np
        else:
            file_np = np.array([((data_output_folder_path), (name),(mass))], dtype=dt_object)
            csv_files_array = np.hstack((csv_files_array,file_np))
            # print('csv_files_array',csv_files_array)
        object_count += 1

    return graph_color_ranges, csv_files_array 