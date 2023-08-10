# Sets up files
# The data folder and sub folders (color_i, color_o etc.)  are not set up automatically. 
# The data folder must be in the same directory as the program
## TODO In the future create the necessary data folder and sub-folders if not made 

import os
from shutil import copyfile

from tracker.lib.user_input import show_files, select_what_to_do_with_files, select_pixel_radius, make_new_folder
from tracker.lib.general import create_data_file
from tracker.lib.color import choose_or_create_color_range
from tracker.lib.infrared import choose_or_create_objects



def create_calc_file(file_path):
    # Creates a file to put trendlines
    with open(file_path, "w") as f:
        f.write("Calculations\n")




    # Make a new folder to save all of the files int 

def set_up_color(src):    
    #### Main Program to setup color tracker ####
    # Generic variables for the color tracker
    # type_of_tracking = 'color'

    file_type = '.npy'
    input_folder = 'color_i'
    data_output = 'color_o'


    print('Welcome to the wonderful world of Color Tracking')

    # Shows the user the color files to choose from
    dir_path_npy= show_files(input_folder, file_type)

    # The user either uses a file, creates a new file, uses a default or erases a file 
    what_to_do_with_npy_files = select_what_to_do_with_files()

    # Selects the range of color for each object
    color_ranges = choose_or_create_color_range(dir_path_npy, what_to_do_with_npy_files, src)

    # Select how small you want the object you track. Having a larger value reduces noise
    min_radius_of_object =  select_pixel_radius()
    max_num_point = len(color_ranges)

    # Make a new folder to save all of the files int

    return file_type, data_output, color_ranges, min_radius_of_object, max_num_point

def set_up_id():    
    #### Main Program to setup color tracker ####
    # Generic variables for the color tracker
    # type_of_tracking = 'infrared' or 'id300'

    file_type = '.npy'
    input_folder = 'infrared_i'
    data_output = 'infrared_o'

    print('Welcome to the wonderful world of Background Subtraction Tracking')

    # Shows the user the color files to choose from
    dir_path_npy= show_files(input_folder, file_type)

    # The user either uses a file, creates a new file, uses a default or erases a file 
    what_to_do_with_npy_files = select_what_to_do_with_files()

    # Selects the range of color for each object
    new_monochrome_ranges= choose_or_create_objects(dir_path_npy, what_to_do_with_npy_files)

    # Select how small you want the object you track. Having a larger value reduces noise
    min_radius_of_object =  select_pixel_radius()
    max_num_point = len(new_monochrome_ranges)

    # Make a new folder to save all of the files int

    return file_type, data_output, new_monochrome_ranges, min_radius_of_object, max_num_point


def make_csv_files(object_ranges, data_output_folder_path):
    print('The data for each object will be stored in this location:')
    if not os.path.exists(data_output_folder_path):
        os.makedirs(data_output_folder_path)
    for (_,_,color_name,_,_) in object_ranges:
        csv_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/'+ color_name + '.csv'))
        create_data_file(csv_file_path)
        print(csv_file_path, '\n')
        

def make_csv_files_no_color(monochrome_ranges, data_output_folder_path):
    print('The data for each object will be stored in this location:')
    for (object_name,_,_) in monochrome_ranges:
        csv_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/'+ object_name + '.csv'))
        create_data_file(csv_file_path)
        print(csv_file_path,'\n')

def clone_csv_files(color_ranges, data_output_folder_path, tracking_run):
    print('The data for each object will be stored in this location:')
    general_spreadsheet = os.path.abspath(os.path.join(data_output_folder_path + '/spreadsheet.csv')) 
    new_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/'+ tracking_run + '/' ))
    for (_,_,color_name,_,_) in color_ranges:
        csv_file_path = os.path.abspath(os.path.join(new_file_path + '/'+ color_name + '.csv'))
        copyfile(general_spreadsheet, csv_file_path)
        print(csv_file_path, '\n')

def create_new_folder (file_path, new_sheet_folder):
    trim_trial = 0
    check = False
    while not check:
        data_output_folder_path = os.path.abspath(os.path.join(file_path , new_sheet_folder , ''))
        # Create a folder to store all of the data in if it does not already exist
        if not os.path.exists(data_output_folder_path):
            os.makedirs(data_output_folder_path)
            check = True
        else:
            trim_trial += 1
            new_sheet_folder = 'trendlines' + str(trim_trial)
    return data_output_folder_path
