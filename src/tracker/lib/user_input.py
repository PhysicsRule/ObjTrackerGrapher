# For the terminal based programs, the user inputs are here.
# This does not include selecting the color file as it is too specific to the color program

import os
import numpy as np
import cv2
import matplotlib.image as mpimg  # Reads images from a folder

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from scipy.signal import lfilter
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

def temp_GUI_select_multiple_files (input_folder):
# select mulitple files is useful when you want to graph many objects
    
    dir_path = show_folders(input_folder)
    file_path, folder_name = select_files_to_graph(dir_path)
    
    return file_path

def select_object_tracker_method():
    # https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
    # when tracking any object, not just a round one, select the method
    # CSRT, KCF , MOSSE are most common
    # CSRT high object accuracy at lower frame rate in real-time
    # KCF faster fps less accuracy
    # MOSSE pure speed

    tracker = None
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
    
    # Create the proper tracker type, depending on the version
    major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.') 

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()

        # high frame rate works for small changes in x,y
        if tracker_type == 'MEDIANFLOW':            
            tracker = cv2.TrackerMedianFlow_create()

        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

        # low frame rate,but quality tracking
        if tracker_type == 'CSRT':                  
            tracker = cv2.TrackerCSRT_create()
    return tracker

def select_type_of_tracking():
# Select if you are using a color or infrared tracking.
# Useful in the graphing program    
    selection_check = False
    while not selection_check:
        type_of_tracking = input('Are you using (c)olor , (i)nfrared tracking or other (o)?')
        if type_of_tracking =='c' or type_of_tracking =='C' or type_of_tracking =='Color':
            type_of_tracking = 'color'
            #input for color hue saturation and darkness
            data_input = 'color_i'
            # output of the position for color tracking
            data_output = 'color_o'
            selection_check = True

        elif type_of_tracking =='i' or type_of_tracking =='I' or type_of_tracking =='Infrared':
            type_of_tracking = 'infrared' 
            data_input = 'infrared_i'
            data_output = 'infrared_o'
            selection_check = True   
        elif type_of_tracking =='o' or type_of_tracking =='O' or type_of_tracking =='other':
            type_of_tracking = 'other' 
            data_input = 'other_i'
            data_output = 'other_o'
            selection_check = True   
            # TODO when encorporated with other trackers this may be useful, so add the folders
            #  Useful for getting data from any source
            print('You chose other, this only works for graphing at the moment')
            print('Make sure you are using .csv files')
    return type_of_tracking, data_input, data_output

def show_folders(input_folder):
# Show the user the file options to choose from
    basepath = os.getcwd()
    dir_path = os.path.abspath(os.path.join(basepath, 'data',input_folder, ''))
    print("These are the folders available =\n", dir_path)
    for f in os.listdir(dir_path):
        print(f)
    return dir_path

def show_files(input_folder, file_type):
# Show the user the file options to choose from
    basepath = os.getcwd()
    dir_path = os.path.abspath(os.path.join(basepath, 'data',input_folder, ''))
    print("These are the files available =\n", dir_path)
    for f in os.listdir(dir_path):
        if f.endswith(file_type):
            print(f)
    return dir_path
    
def select_files_to_graph(dir_path):
# Select the file name
    while True:
        folder_name = input('What is the file/folder you want? \n')
        file_path = os.path.abspath(os.path.join(dir_path, folder_name ))
        if os.path.exists(file_path):
            print('You chose \n', file_path)
            break
        else: 
            print('Name a folder that exists') 
    return file_path, folder_name

def select_multiple_images (type_of_tracking, input_folder, file_type_image):
# puts images into a string to read from
# TODO In the future have the user select the files they want instead of having to delete them from the files themselves

    dir_path = show_folders(input_folder)
    filepath, folder_name = select_files_to_graph(dir_path)
    
    first_frame = int(input('What is the first frame you would like to use? \n'))
    last_frame = int(input('What is the last frame you would like to use? \n'))
    images = []
    for filename in os.listdir(filepath):
        try:
            frame_digit = (str(filename)[-10:-4])
            frame_num = int(frame_digit )
            # if a *.jpg and color/infrared and the frame number is within bounds add it to the array of images
            if type_of_tracking in filename and frame_num>=first_frame and frame_num<=last_frame and file_type_image in filename:
                img = mpimg.imread(os.path.join(filepath, filename))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if img is not None:
                    images.append(img)
        except:
            print('Cant import ' + filename)
    image_array = np.asarray(images)
    return first_frame, last_frame, image_array

def select_multiple_files (input_folder, file_type):
# select mulitple files is useful when you want to graph many objects
    num_files = 1
    dt = np.dtype([ ('filepath', np.str_, 60), ('filename', np.str_, 30)])
    
    
    dir_path = show_folders(input_folder)
    filepath, folder_name = select_files_to_graph(dir_path)
    
    for file_name in os.listdir(filepath):
        if file_name.endswith(file_type):
            print(file_name)
            if num_files == 1:
                file_np = np.array([((filepath), (file_name))], dtype=dt)
                file_array = file_np
                print('filenp ', file_np)
            else:
                file_np = np.array([((filepath), (file_name))], dtype=dt)
                file_array = np.hstack((file_array,file_np))
                print(file_array)
            num_files +=1
    return num_files, file_array

def select_what_to_do_with_files ():
# user chooses if they want to do with the files. This may include making a new color range to track
    selection_check = False
    while not selection_check:
        print('To select your color for your objects... /n')
        print('Do you want to read from a file (f), create a new one (n), erase an old file (e) or use the default(d) ?\n')
        what_to_do_with_npy_files = input('?')
        if what_to_do_with_npy_files == 'f' or what_to_do_with_npy_files == 'n' or what_to_do_with_npy_files == 'e' or what_to_do_with_npy_files == 'd':
            selection_check = True
    return what_to_do_with_npy_files



def select_image_to_read_first(data_output_folder_path, type_of_tracking):
    # select the first frame to start collecting data from
    
    first_image = int(input('What is the frame number of the first image you want to use that has the colored object?'))
    jpg_image = cv2.imread(data_output_folder_path + '/'+ 'image' + str(first_image).zfill(6) +'.jpg')
    return jpg_image, first_image

def select_pixel_radius():
# Select how small you want the object you track to possibly be
    
    print('Set the smallest size object to track. \n')
    print('If you set the radius too small, you will get background objects tracked.')
    print('If you set the radius too large, you the program will not see your object as being big enough.')
    print('If you want to track small objects then set the radius to 10 and you can track a tennis ball 3 meters away.')
    print('If you want to track small objects then set the radius to 20 and you can track a tennis ball 1 meter away.')
    print('If you want to track larger objects then set the radius to 40 and you can track a playground ball 3 meters away.')
    selection_check = False
    while not selection_check:
        try:
            min_radius_of_object = float(input('What do you want to set the smallest pixel radius to be?: '))
            selection_check = True
        except:
             print('Enter a positive number here')  
    return min_radius_of_object
            
def make_new_folder(data_output):
# Make a new folder to save all of the files into
    base_path = os.getcwd()
    dir_path = os.path.abspath(os.path.join(base_path, 'data', data_output , ''))
    print("These are the existing folders.  \n")
    for f in os.listdir(dir_path):
        print(f)
    
    check = False
    while not check:
        data_output_folder = input("What folder do you want to create? (not one of these) \n")
        data_output_folder_path = os.path.abspath(os.path.join(dir_path , data_output_folder , ''))
    
        # Create a folder to store all of the data in if it does not already exist
        if not os.path.exists(data_output_folder_path):
            os.makedirs(data_output_folder_path)
            check = True
        else:
            print('instead of overwriting files, please make a new folder')
        
    print('\nYour folder will be available:\n', data_output_folder_path)
    return data_output_folder, data_output_folder_path

def make_new_folder_trial(data_output_folder_path):
# Make a new folder to save all of the files into
    
    dir_path = os.path.abspath(os.path.join(data_output_folder_path, ''))
   
    check = False
    while not check:
        tracking_run = input("Name this tracking run. This will create a folder. \n")
        tracking_run_folder_path = os.path.abspath(os.path.join(dir_path , tracking_run , ''))
    
        # Create a folder to store all of the data in if it does not already exist
        if not os.path.exists(tracking_run_folder_path):
            os.makedirs(tracking_run_folder_path)
            check = True
        else:
            print('instead of overwriting files, please make a new folder')
        
    print('\nYour folder will be available:\n', tracking_run_folder_path)
    return tracking_run, tracking_run_folder_path

