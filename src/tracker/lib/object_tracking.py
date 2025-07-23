import os

import cv2
import numpy as np

from tracker.lib.user_input import select_object_tracker_method

from tracker.lib.cameras.camera_manager import camera

def GUI_select_bounding_box(pipeline):
    print('select bounding box')
    check_no_selection = True
    while check_no_selection:
        frame_result = camera.get_all_frames_color(pipeline)
        if not frame_result:
            continue
        (cv_color, rs_color, rs_depth), _ = frame_result
        
        ## TODO use rs_infrared to see infrared from camera 1
        depth_image = np.asanyarray(rs_depth.get_data())
        color_image = np.asanyarray(rs_color.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)# Create a colormap from the depth data
        # User inputs the type of tracking used
        tracker = select_object_tracker_method()
        # Select the object to track
        print('select object')
        bbox = cv2.selectROI('ROI Selection', color_image, False)          
        ret = tracker.init(depth_colormap, bbox)
        print(bbox)
        cv2.destroyWindow('ROI Selection')
        check_no_selection = False
    return bbox, ret, tracker

def GUI_select_bounding_box_infrared(pipeline):
    print('select bounding box')
    check_no_selection = True
    while check_no_selection:
        frame_result = camera.get_all_frames_infrared(pipeline)
        if not frame_result:
            continue
        (rs_depth, rs_infrared1), _ = frame_result
        
        ## TODO use rs_infrared to see infrared from camera 1
        depth_image = np.asanyarray(rs_depth.get_data())
        infrared_image = np.asanyarray(rs_infrared1.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)# Create a colormap from the depth data
        # User inputs the type of tracking used
        tracker = select_object_tracker_method()
        # Select the object to track
        print('select object')
        bbox = cv2.selectROI('ROI Selection', infrared_image, False)          
        ret = tracker.init(depth_colormap, bbox)
        print(bbox)
        cv2.destroyWindow('ROI Selection')
        check_no_selection = False
    return bbox, ret, tracker

def draw_bounding_box(image, bounding_box):
    # Draws a box to indicate what object is being tracked and noting its starting position.
    x_coordinate = int(bounding_box[0])
    y_coordinate = int(bounding_box[1])
    width = int(bounding_box[2])
    height = int(bounding_box[3])
    cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (255, 0,0), 3,1)

def find_xy_using_tracking_method(tracker, bbox, cv_image):
    # Updtates the bbox using the tracker method chosen
    ret, bbox = tracker.update(cv_image)
    if ret:
        # print('found a frame')# TODO add multiple objects and have a count of 0 and 1 so 2 objects
        x_pixel = int(bbox[0]+bbox[2]/2)
        y_pixel = int(bbox[1]+bbox[3]/2)
        radius_of_bbox = int(bbox[2]/2) # plots a circle around object instead of a box
    else:
        return -1, -1, None, 0 
        print('no object found')
    return x_pixel, y_pixel, bbox, radius_of_bbox 


