## Color Tracker
## Track objects by color using Intel RealSense D435i.
## The largest object of the color band chosen will be tracked by it's center.
## Infrared tracking by subtracting the original scene was done in the past, so components of these features are still present.

import os
import cv2
import numpy as np

from tracker.lib.setup_files import set_up_color, make_csv_files
from tracker.lib.intel_realsense_D435i import get_all_frames_color, get_all_frames_infrared, get_depth_meters, find_and_config_device, select_furthest_distance_color, select_furthest_distance_infrared, warm_up_camera
from tracker.lib.color import make_color_hsv, find_object_by_color
from tracker.lib.general import save_video_file 
from tracker.lib.color import GUI_find_hsv_bounds
from tracker.lib.object_tracking import GUI_select_bounding_box, GUI_select_bounding_box_infrared, find_xy_using_tracking_method, draw_bounding_box


def find_lower_upper_bounds_on_screen(the_array):
    # selecting your own color boundaries by tweeking the default
    pipeline = find_and_config_device()
    warm_up_camera(pipeline)
    (cv_color, rs_color, rs_depth), timestamp = get_all_frames_color(pipeline)

    output = GUI_find_hsv_bounds(the_array, cv_color)
    return output

def what_to_do_with_images(i, x_pixel, y_pixel, radius,image,depth_colormap,cv_color,mask,data_output_folder_path):
    # shows the object on the images chosen
    # saves the images chosen into the data folder
    
    if image.show_depth:
        # Show depth colormap & color feed
        cv2.circle(depth_colormap, (int(x_pixel), int(y_pixel)), int(radius), (255, 255, 255), 2)  
        cv2.imshow('depth', depth_colormap)
        cv2.moveWindow('depth',850,0)
    if image.show_RGB:
        cv2.circle(cv_color, (int(x_pixel), int(y_pixel)), int(radius), (255, 255, 255), 2)  
        cv2.imshow('Tracking', cv_color)
        cv2.moveWindow('Tracking',0,0)
    
    if image.show_mask and mask is not None:
        cv2.imshow('mask', mask)
        cv2.moveWindow('mask',0,500)
    
    # Save the RGB and depth images to view later if you want, but it does slow the tracking down a bit.
    if image.save_RGB:
        color_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'color'+  str(i) + '.jpg'))   
        cv2.imwrite(color_file_path,cv_color)
    '''Saving Depth is now just saving the video
    if image.save_depth:
        depth_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'depth'+  str(i) + '.jpg'))   
        cv2.imwrite(depth_file_path, depth_colormap)'''
    if image.save_mask:
        mask_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'mask'+  str(i) + '.jpg'))   
        img_mask = cv2.imread(mask)
        if img_mask is None:
            print("Error: Image not loaded correctly")
        else:
            cv2.imwrite(mask_file_path, mask)
    



def GUI_tracking(pipeline, image, color_ranges, min_radius_object, data_output_folder_path, tracking_info):
    """
    Track as you record
    use color or obj_tracking
    Hit graph afterwards to see results
    about 30 fps when alining color image to depth image
    """ 
    make_csv_files(color_ranges, data_output_folder_path)   
    max_num_point=len(color_ranges)
    warm_up_camera(pipeline)
    zeroed_x, zeroed_y, zeroed_z, clipping_distance = select_furthest_distance_color(pipeline)
    if tracking_info.type_of_tracking == 'obj_tracker':
        print('select bounding box')
        bbox, ret, tracker = GUI_select_bounding_box(pipeline)    

    # Now that everything is setup, track the objects
    first_time_check = True
    image_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/video/'))  
    video_RGB_array = []
    video_depth_array = []
    start_time = 0 # It should get a time the first round through
    i=0

    x_pixel, y_pixel, x_coord, y_coord, z_coord, radius  = -1, -1, -1, -1, -1, 0
    while True:
        # Get frames if valid
        frame_result = get_all_frames_color(pipeline)
        if not frame_result:
            continue        
        (cv_color, rs_color, rs_depth), timestamp = frame_result

        # Start the timer
        if first_time_check:
            start_time = timestamp
            # we might want this later and compare with start that has milliseconds
            first_time_check = False

            # Converts time from milliseconds to seconds
        relative_timestamp = round(((timestamp - start_time) / 1000),3)
            
        ## Color Tracking by making a mask for each color tracked
        hsv = make_color_hsv(cv_color)
        depth_image = np.asanyarray(rs_depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)# Create a colormap from the depth data

        for (lower,upper, color_name, radius_meters, mass) in color_ranges:
            # Find location of the object in x,y pixels using color masks
            mask = None
            x_pixel, y_pixel, x_coord, y_coord, z_coord, radius  = -1, -1, -1, -1, -1, 0
            if tracking_info.type_of_tracking == 'color':
                x_pixel, y_pixel, radius, mask_limited_array = find_object_by_color(cv_color,hsv, lower,upper, color_name, radius_meters, mass, min_radius_object, max_num_point)     
                if mask_limited_array is None:
                    continue
                mask =np.array(mask_limited_array)
            elif tracking_info.type_of_tracking == 'obj_tracker':
                cv_image= depth_colormap
                radius = 0 # showing if tracker is not working
                mask = None
                x_pixel, y_pixel, bbox, radius = find_xy_using_tracking_method(tracker, bbox, cv_image)
                draw_bounding_box(cv_image, bbox)

            if x_pixel == -1 or y_pixel == -1:
                continue
            # get.distance is a little slower so only use if necessarycenter = round(aligned_depth_frame.get_distance(int(x),int(y)),4)

            x_coord, y_coord, z_coord = get_depth_meters(x_pixel, y_pixel, radius_meters, rs_depth, rs_color, zeroed_x, zeroed_y, zeroed_z, clipping_distance)
            if x_coord == -1 or y_coord == -1:
                continue
            # Append to the file until there is an error at which it will close

            # Writes the coordinates to each colored object
            csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, color_name + '.csv'))   
            with open(csv_file_path, 'a') as data_to_file:
                data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n')      

            if color_name == color_ranges[0][2]:
                for image_to_show in (cv_color,depth_colormap, mask):
                    cv2.putText(image_to_show, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                    cv2.putText(image_to_show, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                    cv2.putText(image_to_show, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                    cv2.putText(image_to_show, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            
        what_to_do_with_images(i, x_pixel, y_pixel, radius,image,depth_colormap,cv_color,mask,data_output_folder_path)
        
        # saves the images into an array to convert to a movie when stopped
        if image.save_video:
            video_RGB_array.append(cv_color)
        if image.save_depth:
            video_depth_array.append(depth_colormap)

        i +=1

        # exit if spacebar or esc is pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == 32:
            print('end')
            # Close all OpenCV windows
            
            cv2.destroyAllWindows()
            break

    # Cleanup after loop break

    # Stop OpenCV/RealSense video

    pipeline.stop()

    if image.save_video:
        height, width, layers = cv_color.shape
        size = (width,height)
        save_video_file(image_file_path, video_RGB_array,'RGB', size)

    if image.save_depth:
        height, width, layers = depth_colormap.shape
        size = (width,height)
        save_video_file(image_file_path, video_depth_array, 'Depth', size)
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()


def GUI_obj_tracking(pipeline, image, color_ranges, min_radius_object, data_output_folder_path, tracking_info):
    """
    Track as you record
    use obj_tracking
    Hit graph afterwards to see results
    about no need to align
    uses infrared1 and depth instead of color
    """ 
    make_csv_files(color_ranges, data_output_folder_path)   
    max_num_point=len(color_ranges)
    warm_up_camera(pipeline)
    zeroed_x, zeroed_y, zeroed_z, clipping_distance = select_furthest_distance_infrared(pipeline)
    print('select bounding box')
    bbox, ret, tracker = GUI_select_bounding_box_infrared(pipeline)    

    # Now that everything is setup, track the objects
    first_time_check = True
    image_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/video/'))  
    video_img_array = []
    video_depth_array = []
    start_time = 0 # It should get a time the first round through
    i=0
    x_pixel, y_pixel, x_coord, y_coord, z_coord, radius  = -1, -1, -1, -1, -1, 0

    while True:
        # Get frames if valid
        frame_result = get_all_frames_infrared(pipeline)
        if not frame_result:
            continue        
        (rs_depth, rs_infrared), timestamp = frame_result
        
        # Start the timer
        if first_time_check:
            start_time = timestamp
            # we might want this later and compare with start that has milliseconds
            first_time_check = False

        # Converts time from milliseconds to seconds
        relative_timestamp = round(((timestamp - start_time) / 1000),3)
        
        ## TODO use rs_infrared to see infrared from camera 1
        depth_image = np.asanyarray(rs_depth.get_data())
        infrared_image = np.array(rs_infrared.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)# Create a colormap from the depth data
        # image the user sees as if it were color
        cv_color = infrared_image    
        ## Color Tracking by making a mask for each color tracked

        for (lower,upper, color_name, radius_meters, mass) in color_ranges:
        # tracking_info.type_of_tracking = 'obj_tracker'
            x_pixel, y_pixel, x_coord, y_coord, z_coord, radius  = -1, -1, -1, -1, -1, 0
            mask = None
            x_pixel, y_pixel, bbox, radius = find_xy_using_tracking_method(tracker, bbox, depth_colormap)
            
            if x_pixel == -1:
                continue
            # get.distance is a little slower so only use if necessarycenter = round(aligned_depth_frame.get_distance(int(x),int(y)),4)

            x_coord, y_coord, z_coord = get_depth_meters(x_pixel, y_pixel, radius_meters, rs_depth, rs_infrared, zeroed_x, zeroed_y, zeroed_z, clipping_distance)
            if x_coord == -1:
                continue
            # Append to the file until there is an error at which it will close

            # Writes the coordinates to each colored object
            csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, color_name + '.csv'))   
            with open(csv_file_path, 'a') as data_to_file:
                data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n')      

            if color_name == color_ranges[0][2]:
                for image_to_show in (cv_color,depth_colormap):
                    cv2.putText(image_to_show, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                    cv2.putText(image_to_show, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                    cv2.putText(image_to_show, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                    cv2.putText(image_to_show, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                # draw_bounding_box(cv_color, bbox)
        what_to_do_with_images(i, x_pixel, y_pixel, radius,image,depth_colormap,cv_color,mask,data_output_folder_path)
        
        # saves the images into an array to convert to a movie when stopped
        if image.save_video:
            video_img_array.append(cv_color)
        if image.save_depth:
            video_depth_array.append(depth_colormap)
        i +=1

        # exit if spacebar or esc is pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == 32:
            print('end')
            # Close all OpenCV windows
            
                
            cv2.destroyAllWindows()
            break

    # Cleanup after loop break

    # Stop OpenCV/RealSense video

    pipeline.stop()
    if image.save_video:
        # Since cv_color is really an infrared image it does not have layers
        height, width = cv_color.shape
        size = (width, height)
        save_video_file(image_file_path, video_img_array,'img', size)

    if image.save_depth:
        height, width, layers = depth_colormap.shape
        size = (width,height)
        save_video_file(image_file_path, video_depth_array, 'Depth', size)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
