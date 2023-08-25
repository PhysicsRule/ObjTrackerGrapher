import os

import cv2
import numpy as np
import pyrealsense2 as rs
import imutils
import time

from tracker.lib.intel_realsense_D435i import get_all_frames_color
from tracker.lib.user_input import select_object_tracker_method

def GUI_select_bounding_box(pipeline):
    print('select bounding box')
    check_no_selection = True
    print('find the bbox')
    while check_no_selection:
        frame_result = get_all_frames_color(pipeline)
        if not frame_result:
            continue
        (cv_color, rs_color, rs_depth), _ = frame_result
        # User inputs the type of tracking used
        tracker = select_object_tracker_method()
        # Select the object to track
        print('select object')
        bbox = cv2.selectROI('ROI Selection', cv_color, False)          
        ret = tracker.init(cv_color, bbox)
        cv2.destroyWindow('ROI Selection')
        check_no_selection = False
    return bbox, ret

def find_xyz_using_tracking_method():
    print('find object xy using tracker')

# Show the user the first image from the bag file you want to read.
jpg_image, first_image = select_image_to_read_first(data_output_folder_path, type_of_tracking)

# The image file is the cv_color when using color tracking
cv_color = jpg_image

# Find the furthest distance and in the future find a different origin TODO
#zeroed_x, zeroed_y, zeroed_z, clipping_distance = select_furthest_distance_color(cv_video, pipeline)



    # if type_of_tracking == 'back':
    show_image = True
    save_image = True
    first_time_check = True
    start_time = 0 # It should get a time the first round through
    save_depth_image = True
    timestamp = 0.0000
    zeroed_x, zeroed_y, zeroed_z = 0.0, 0.0, 0.0

while continue_reading is True:
    i += 1
    try:
        
        rs_frames = pipeline.wait_for_frames()
        timestamp = rs_frames.get_timestamp()
        print("Reading and frame:", i, timestamp)
        if not rs_frames:
            print("couldn't read", i)
            if i>6000:
                pipeline.stop()
                continue_reading = False
            continue
        
        # Start the timer
        if first_time_check:
            start_time = timestamp
            # we might want this later and compare with start that has miliseconds
            first_time_check = False
        # Converts time from miliseconds to seconds
        relative_timestamp = (timestamp - start_time) / 1000
        print (timestamp,'    ',relative_timestamp)
        
        if types_of_streams_saved =='cd': 
            # Color frames
            rs_align = rs.align(rs.stream.color)
            rs_frames_aligned = rs_align.process(rs_frames)

            # Extract color/depth frames
            rs_color = rs_frames_aligned.get_color_frame()
            rs_depth = rs_frames_aligned.get_depth_frame().as_depth_frame()
            # Create depth numpy array
            depth_image = np.asanyarray(rs_depth.get_data())
            # TODO erase if works 
            # rs_color = rs_frames.get_color_frame()
            cv_color = np.array(rs_color.get_data())
            #cv2.imwrite(folderPath + "/color"+ str(i).zfill(6) +  ".png", color_image)
            x_coord, y_coord, z_coord = find_Time_x_y_z_from_rs_frames(i, save_image, show_image, cv_color, rs_color, rs_depth, relative_timestamp, data_output_folder_path, object_ranges, zeroed_x, zeroed_y, zeroed_z, clipping_distance, min_radius_of_object, max_num_point)
            
            # Show the timestamp on the color image
            #output(i, save_image, 'color' , data_output_folder_path, cv_color, relative_timestamp, x_coord, y_coord, z_coord, object_ranges)


        elif 'id' in types_of_streams_saved: 
            rs_depth = rs_frames.get_depth_frame()
            depth_image = np.asanyarray(rs_depth.get_data())
            #infrared frames
            rs_infrared = rs_frames.get_infrared_frame(1)
            print ('got it', timestamp)

            #If using the depth instead of the monochrome color from infrared image
            if type_of_tracking =='backd':
                image = depth_image
  
            else:
                image = np.array(rs_infrared.get_data())
            if type_of_tracking == 'obj_tracker':
                ret, bbox = tracker.update(image)
                if ret:
                    print('found a frame')
                    find_object_by_tracker(bbox, i, relative_timestamp, backgd, image, rs_depth, rs_infrared, data_output_folder_path, type_of_tracking, zeroed_x, zeroed_y, zeroed_z, clipping_distance, object_ranges, max_num_point)
            else: 
                try:
                    x_coord, y_coord, z_coord = find_object_by_subtracting_background(i, relative_timestamp, backgd, image, rs_depth, rs_infrared, data_output_folder_path, type_of_tracking, zeroed_x, zeroed_y, zeroed_z, clipping_distance, object_ranges, min_radius_of_object, max_num_point)
                except Exception as e:
                    print ('Failed to get x_coord from background subtraction',e)
                    x_coord, y_coord, z_coord = None, None, None
        
            #cv2.imwrite(folderPath + "/infrared"+ str(i).zfill(6) +  ".png", infrared_image)

        
        

            
        # Show colormap of depth image and write it to a file
        #if save_depth_image == True:
        #    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)
        #    output(i, save_image, 'depth' , data_output_folder_path, depth_colormap,relative_timestamp, x_coord, y_coord, z_coord, None)
        # Based on the type of tracking method selected, run different parts of other code.
        # if type_of_tracking == 'back':
        

        # exit if spacebar or esc is pressed
        k = cv2.waitKey(1) & 0xff
        if k ==27 or k==32:
            pipeline.stop()
            print ('end')
            break
    
    
    except RuntimeError:
        print("There are no more frames left in the .bag file!")
        pipeline.stop()
        continue_reading = False
        break
    finally:
        pass
    


#------------------------------------
  # https://stackoverflow.com/questions/58482414/frame-didnt-arrived-within-5000-while-reading-bag-file-pyrealsense2  


#        playback.resume()
#
#except RuntimeError:
#    print("There are no more frames left in the .bag file!")
#


