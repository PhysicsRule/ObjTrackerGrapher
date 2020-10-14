## Create a npy color range from a frame
## It should also be able to be used to select the frames you want to use to collect the data from, but it is not working right now
## DOESN"T WORK unless images are named and named in alphabetical and numerical order##
## You should be able to select the first and last frame to look through, but it has trouble reading them.
## There may be other errors ##

import numpy as np

from tracker.lib.user_input import select_multiple_images, show_files, show_folders, select_files_to_graph, select_what_to_do_with_files, select_pixel_radius, make_new_folder
from tracker.lib.general import create_data_file
from tracker.lib.color import choose_or_create_color_range, read_hsv_bounds
   
file_type = '.npy'
type_of_tracking = 'color'
data_folder = 'color_i'
data_output = 'color_o'


# Select folder to read images from
# type_of_tracking, _, data_output = select_type_of_tracking() 
print('Delete images from the file you will not use to make it easier to select the color more easily')
#type_of_tracking, _, data_output = select_type_of_tracking() 

# put all of the jpg images in an array
file_type_image = '.jpg'
first_frame, last_frame, image_array = select_multiple_images(type_of_tracking, data_output, file_type_image)


# Shows the user the color files to choose from
dir_path_npy= show_files(data_folder, file_type)

# The user will make a new(n) *npy file
what_to_do_with_npy_files = 'n'
src = image_array  
i, new_color_ranges = read_hsv_bounds(src)
color_ranges = np.array(new_color_ranges)
print(color_ranges)
selection_check_what_to_do = True

