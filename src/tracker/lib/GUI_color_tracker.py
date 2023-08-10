## Color Tracker
## Track objects by color using Intel RealSense D435i.
## The largest object of the color band chosen will be tracked by it's center.
## Infrared tracking by subtracting the original scene was done in the past, so components of these features are still present.

import os
import argparse
import cv2
import numpy as np

from tracker.lib.setup_files import set_up_color, make_csv_files
from tracker.lib.intel_realsense_D435i import get_all_frames_color, get_depth_meters, find_and_config_device, select_furthest_distance_color, warm_up_camera
from tracker.lib.color import make_color_hsv, find_object_by_color
from tracker.lib.general import open_the_video 
from tracker.lib.color import GUI_find_hsv_bounds

def find_lower_upper_bounds_on_screen(the_array):
    print('selecting colors')
    pipeline = find_and_config_device()
    warm_up_camera(pipeline)
    (cv_color, rs_color, rs_depth), timestamp = get_all_frames_color(pipeline)

    output = GUI_find_hsv_bounds(the_array, cv_color)
    return output


def GUI_color_tracking(pipeline, src, type_of_tracking, image,color_ranges, min_radius_of_object, data_output_folder_path, input_folder, data_output ):
    
    make_csv_files(color_ranges, data_output_folder_path)
    
    max_num_point=len(color_ranges)
    ## GUI TODO Have GUI real existing folders and provide an optional name for a new folder name being the last one listed with the number incremented by 1
    

    # Configure and setup the cameras
    
    warm_up_camera(pipeline)
    # OpenCV initialization


    # Find the furthest distance and in the future find a different origin TODO
    zeroed_x, zeroed_y, zeroed_z, clipping_distance = select_furthest_distance_color(pipeline)

    # Now that everything is setup, track the objects
    first_time_check = True
    image_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/video/'))  
    video_img_array = []
    start_time = 0 # It should get a time the first round through
    i=0

    while True:
        # Get frames if valid
        frame_result = get_all_frames_color(pipeline)
        if not frame_result:
            continue
        
        (cv_color, rs_color, rs_depth), timestamp = frame_result
            
        ## Color Tracking by making a mask for each color tracked
        hsv = make_color_hsv(cv_color)

        for (lower,upper, color_name, radius_meters, mass) in color_ranges:
            # Find location of the object in x,y pixels using color masks
            x_pixel, y_pixel, radius, mask = find_object_by_color(cv_color,hsv, lower,upper, color_name, radius_meters, mass, min_radius_of_object, max_num_point)     

            if x_pixel is None:
                continue
            # get.distance is a little slower so only use if necessarycenter = round(aligned_depth_frame.get_distance(int(x),int(y)),4)

            x_coord, y_coord, z_coord = get_depth_meters(x_pixel, y_pixel, radius_meters, rs_depth, rs_color, zeroed_x, zeroed_y, zeroed_z, clipping_distance)
            if x_coord is None:
                continue
            # Append to the file until there is an error at which it will close

            # Start the timer
            if first_time_check:
                start_time = timestamp
                # we might want this later and compare with start that has milliseconds
                first_time_check = False

            # Converts time from milliseconds to seconds
            relative_timestamp = (timestamp - start_time) / 1000

            # Writes the coordinates to each colored object
            csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, color_name + '.csv'))   
            with open(csv_file_path, 'a') as data_to_file:
                data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n')      

            if color_name == color_ranges[0][2]:
                cv2.putText(cv_color, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(cv_color, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(cv_color, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(cv_color, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            
        depth_image = np.asanyarray(rs_depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)# Create a colormap from the depth data
        if image.show_depth:
            # Show depth colormap & color feed
            cv2.circle(depth_colormap, (int(x_pixel), int(y_pixel)), int(radius), (255, 255, 255), 2)
            cv2.imshow('depth', depth_colormap)
            cv2.moveWindow('depth',850,0)
        if image.show_RGB:
            cv2.imshow('Tracking', cv_color)
            cv2.moveWindow('Tracking',0,0)
        
        if image.show_mask and mask is not None:
            cv2.imshow('mask', mask)
            cv2.moveWindow('mask',0,500)
        
        # Save the RGB and depth images to view later if you want, but it does slow the tracking down a bit.
        if image.save_RGB:
            color_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'color'+  str(i) + '.jpg'))   
            cv2.imwrite(color_file_path,cv_color)
        if image.save_depth:
            depth_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'depth'+  str(i) + '.jpg'))   
            cv2.imwrite(depth_file_path, depth_colormap)
        if image.save_mask:
            mask_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'mask'+  str(i) + '.jpg'))   
            cv2.imwrite(mask_file_path, mask)
        if image.save_video:
            height, width, layers = cv_color.shape
            size = (width,height)
            video_img_array.append(cv_color)

        i +=1

        # exit if spacebar or esc is pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == 32:
            print('end')
            # Close all OpenCV windows
            if image.save_video:
                out = cv2.VideoWriter(image_file_path +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
                for i in range(len(video_img_array)):
                    out.write(video_img_array[i])
                out.release()
            cv2.destroyAllWindows()
            break

    # Cleanup after loop break

    # Stop OpenCV/RealSense video

    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
