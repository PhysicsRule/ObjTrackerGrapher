# Set the color bounds used in tracking.
# Predefined colors set here.
# TODO in the future, make this editable by the user so they have their own predefined color ranges (bounds).
# Gets the depth at the center of the colored object.
# Started with Pyimagesearch code initially

import os
import ast
import cv2
import numpy as np
import pyrealsense2 as rs
import imutils
import time
from math import floor
from typing import Tuple, Optional, Any

from tracker.lib.intel_realsense_D435i import get_depth_meters

def nothing(x):
    pass



def GUI_find_hsv_bounds(the_array, src) -> Optional[np.ndarray]:
    # Find the color range of each object
    # print('the_array', the_array)
    [(lower, upper, color, radius_meters, mass)] = the_array
    real_time = False
    i=0 # frame 
    # Initializing the webcam feed.
    if type(src) == int:
        cap = cv2.VideoCapture(src)
        cap.set(3,640)
        cap.set(4,480)
    else:
        # TODO If the webcam doesn't work have another option to change camera
        frame = src #The rs_color frame as a numpy array

    # Create a window named Press (s) when done.
    cv2.namedWindow("Press (s) when done")
    cv2.putText(frame, 'bounds that show ' + color, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    # Now create 6 Press (s) when done that will control the lower and upper range of 
    # H,S and V channels. The Arguments are like this: Name of trackbar, 
    # window name, range,callback function. For Hue the range is 0-179 and
    # for S,V its 0-255.
    ([l_h, l_s, l_v]) = lower
    ([u_h, u_s, u_v]) = upper
    cv2.createTrackbar("L - H", "Press (s) when done", l_h, 179, nothing)
    cv2.createTrackbar("L - S", "Press (s) when done", l_s, 255, nothing)
    cv2.createTrackbar("L - V", "Press (s) when done", l_v, 255, nothing)
    cv2.createTrackbar("U - H", "Press (s) when done", u_h, 179, nothing)
    cv2.createTrackbar("U - S", "Press (s) when done", u_s, 255, nothing)
    cv2.createTrackbar("U - V", "Press (s) when done", u_v, 255, nothing)

    output = None

    while True:
        # Start reading the webcam feed frame by frame.
        if isinstance(src, int):
            ret, frame = cap.read()
            real_time = True
            if not ret:
                break
        else:
            try:
                if isinstance(frame, np.ndarray):
                    frame = src #The rs_color frame as a numpy array
            except NameError:
                break

        # Flip the frame horizontally (Not required)
        #frame = cv2.flip( frame, 1 ) 
    
        # Convert the BGR image to HSV image.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get the new values of the trackbar in real time as the user changes 
        l_h = cv2.getTrackbarPos("L - H", "Press (s) when done")
        l_s = cv2.getTrackbarPos("L - S", "Press (s) when done")
        l_v = cv2.getTrackbarPos("L - V", "Press (s) when done")
        u_h = cv2.getTrackbarPos("U - H", "Press (s) when done")
        u_s = cv2.getTrackbarPos("U - S", "Press (s) when done")
        u_v = cv2.getTrackbarPos("U - V", "Press (s) when done")
        # Set the lower and upper HSV range according to the value selected
        # by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
    
        # Filter the image and get the binary mask, where white represents 
        # your target color
        mask = cv2.inRange(hsv, lower_range, upper_range)
        # You can also visualize the real part of the target color (Optional)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # Converting the binary mask to 3 channel image, this is just so 
        # we can stack it with the others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # stack the mask, orginal frame and the filtered result
        stacked = np.hstack((mask_3,frame,res))
        # Show this stacked frame at 40% of the size.
        cv2.imshow('Press (s) when done',cv2.resize(stacked,None,fx=0.8,fy=0.8))
    
        # If the user presses ESC then exit the program
        key = cv2.waitKey(1)
        if key == 27:
            # TODO: If this happens, the program may crash
            break
        if key ==ord('>'):
            i+=1
        elif key ==ord('<'):
            i-=1

        if key == ord('s'):
            lower_tuple = (l_h, l_s, l_v)
            upper_tuple = (u_h, u_s, u_v)
            output = np.array( [((lower_tuple),(upper_tuple),color,(radius_meters),(mass))], dtype=object)
            break

    # Release the camera & destroy the windows.
    if real_time:
        cap.release()
    cv2.destroyAllWindows()
    print('output', output)
    return output



def read_hsv_bounds(src):
# Determine which objects and how many objects you are going to track.
# Call a function to find their upper and lower bounds on the color hsv scale
    print('\nTO SELECT A single object \n \b')
    print('-Move the L-H (lower bounds for Hue) skidder until your object just starts to disappear.')
    print('-This is essentially the color')
    print('-then move the U-H (upper bounds for Hue) the other way. The mask on the right is very helpful')
    print('-Repeat this for the saturation of corlor and the brightness of color')

    num_objects = int(input('How many objects do you want to enter?'))
    for i in range(num_objects):
        object_name = input('what do you want to call the object? Examples:  yellowTennisBall , greenball ')
        
        print ("Realize the depth camera will measure the object from the front of the object then in the amount you put for the radius in meters. /n")
        radius_meters = input("What is the radius in meters of the object? enter 0.0 if the object is flat and parallel to camera,")

        mass = input("What is the mass of the object? enter 0.0 if you don't care \n")
        
        thearray = find_hsv_bounds(object_name, radius_meters, mass, src)
        if i==0:
            new_color_ranges = thearray
        else:
            new_color_ranges = np.hstack((new_color_ranges,thearray))          
            
        print(new_color_ranges,' \n')
    
    # Save array for later
    print ('What do you want to call the scene (name of file) that has each object information? For example LoriDining4Balls.')
    name_of_array = input('Just hit enter if you want it to be called, testscene.')
    
    if name_of_array == '':
        name_of_array = 'testscene'

    basepath = os.getcwd()
    npy_file = os.path.abspath(os.path.join(basepath, 'data/color_i/' + name_of_array))
    np.save(npy_file, new_color_ranges)
    
  
    return i, new_color_ranges


def choose_or_create_color_range(dir_path, file_new_delete, src):
    selection_check_what_to_do = False
    while not selection_check_what_to_do:
        if file_new_delete == 'd':
                # If people want the default values (d)
                green_lower = (30, 67, 57)
                green_upper = (85, 255, 188)
                red_1_lower = (0, 20, 6) 
                red_1_upper = (20, 255, 230)
                red_2_lower = (160, 20, 6)
                red_2_upper = (179, 255, 230)
                blue_lower = (58, 71, 52)
                blue_upper = (125, 255, 170)
                yellow_lower = (17, 24, 171)
                yellow_upper = (30, 199, 255)

                color_ranges = [
                    (yellow_lower,yellow_upper, "yellow",0.01,0.0),
                    (green_lower, green_upper, "green",0.01,0.0),
                    (blue_lower,blue_upper, "blue",0.01,0.0)]
                selection_check_what_to_do = True
        elif file_new_delete == 'e':
            remove_file = input('which file do you want to remove? \n')
            os.path.abspath(os.remove(dir_path + '/' + remove_file))
        elif file_new_delete == 'n':
            i, new_color_ranges = read_hsv_bounds(src)
            color_ranges = np.array(new_color_ranges)
            print(color_ranges)
            selection_check_what_to_do = True
        elif file_new_delete == 'f':
        # This means you want a file that you already made
            selection_check_file_name = False
            while not selection_check_file_name:
                np_file = input("Which file do you want?  leave off the *.npy. If you want testscene just hit enter \n")
                if np_file =='':
                    np_file ='testscene'
                np_file_name = str(np_file + '.npy')
                try:
                    npy_file = os.path.abspath(os.path.join(dir_path + '/'+ np_file_name))
                    color_ranges= np.load(npy_file)
                    print(color_ranges)
                    selection_check_file_name = True
                    selection_check_what_to_do = True
                except:
                    print('We could not find that file. Copy and paste to get spelling correct')
        else:
            selection_check_what_to_do =False
            print('\n\n What would you like to do?')
    
    return color_ranges

def make_color_hsv(frame):
    #Blurs image and creates an hsv color frame of image that will be the same for all colors

    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    return hsv

def find_object_by_color(frame, hsv, lower,upper, color_name, radius_meters, mass, min_radius_of_object, max_num_point) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[Any]]:
    # construct a mask for the color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask

    mask = cv2.inRange(hsv, lower,upper)
    ## TODO create a mask that has less saturation than the original to reduce the trail of blurred ball
    mask_in_Range = mask
   
    #if color_options[color] is not 'red':
    #    mask = cv2.inRange(hsv, colormap[color][0], colormap[color][1])
    #else:
    #    mask1 = cv2.inRange(hsv, colormap[color][0], colormap[color][1])
    #    mask2 = cv2.inRange(hsv, colormap[color][2], colormap[color][3])
    #    mask = mask1 | mask2

    # cleans up the mask to show the object more clearly
    mask = cv2.erode(mask, None, iterations=2)
    mask_erode = mask
    mask = cv2.dilate(mask, None, iterations=4)
    mask_dilate = mask

    # find contours in the mask and initialize the current
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
 
    # only proceed if at least one contour was found
    if len(contours) > 0:
	    # find the largest contour in the mask, then use
	    # it to compute the minimum enclosing circle and centroid

        contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
        max_index = contour_areas.argsort()[::-1][0]
        largest_contour = contours[max_index]
        ((x, y), pixel_radius) = cv2.minEnclosingCircle(largest_contour)
        if x is None: 
            x=-1
            pixel_radius = 0    
        # Draw a circle around the contour it found then print the mask
        # This is for testing what you found. 
        if x != -1:
            cv2.circle(mask, (int(x), int(y)), int(pixel_radius+5), (255,255,255), 2)

        # only proceed if the radius meets a minimum size
        if pixel_radius > min_radius_of_object:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(pixel_radius), (0,0,0), 2)
            ## TODO add back in after the paper is done
            # cv2.putText(frame, color_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2);

        return x, y, pixel_radius, mask
    else:
        return -1, -1, 0, None

def find_object_by_color_with_red(cv_color, color, color_ranges) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[Any]]:

    blurred = cv2.GaussianBlur(cv_color, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if color != 'double_red':
        mask = cv2.inRange(hsv, color_ranges[0], color_ranges[1])
    else:
        mask1 = cv2.inRange(hsv, color_ranges[0], color_ranges[1])
        mask2 = cv2.inRange(hsv, color_ranges[2], color_ranges[3])
        mask = mask1 | mask2

    # Erode & dilate mask to remove small blobs
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    # Extract contours from image
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inspect contours if they exist
    if contours:
        # Just grab largest contour for now; multiple could be done later
        contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
        max_index = contour_areas.argsort()[::-1][0]
        largest_contour = contours[max_index]

        ((x, y), pixel_radius) = cv2.minEnclosingCircle(largest_contour)
    
        return x, y, pixel_radius, mask
    else:
        return None, None, None, None

def find_Time_x_y_z_from_rs_frames(i,save_image, show_image, cv_color, rs_color, rs_depth, relative_timestamp, data_output_folder_path, color_ranges, zeroed_x, zeroed_y, zeroed_z, clipping_distance, min_radius_of_object, max_num_point):
    ###TODO setup the trailing image so this will work for other data as well.
    save_trailing_image = False
    if save_trailing_image:
        trailing_color = cv2.imread(data_output_folder_path + '/'+ 'trailing'+'.jpg')

    ## TODO change this to save depth image and show depth image
    if save_image:
        depth_image = np.asanyarray(rs_depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)

    hsv = make_color_hsv(cv_color)
    x_pixel = None
    for (lower,upper, color_name, radius_meters, mass) in color_ranges:
        # Find location of the object in x,y pixels using color masks
        x_pixel, y_pixel, pixel_radius, mask = find_object_by_color(cv_color,hsv, lower,upper, color_name, radius_meters, mass, min_radius_of_object, max_num_point)     

        if x_pixel is None:
            continue
        # get.distance is a little slower so only use if necessarycenter = round(aligned_depth_frame.get_distance(int(x),int(y)),4)
        
        # Get the distance using the points around the center of the object
        depth_near_center = []
        for pixel_count in range(-1,2):
            _, _, z_coord = get_depth_meters(x_pixel+pixel_count, y_pixel+pixel_count, radius_meters, rs_depth, rs_color, zeroed_x, zeroed_y, zeroed_z, clipping_distance)
            if z_coord is not None:
                depth_near_center.append(z_coord)
        # Median for our 5 points
        depth_near_center.sort()
        if len(depth_near_center) == 0:
            x_coord, y_coord, z_coord = None, None, None
            continue
        
        # Get the x and y_coord at the exact center of the object
        x_coord, y_coord, _ = get_depth_meters(x_pixel, y_pixel, radius_meters, rs_depth, rs_color, zeroed_x, zeroed_y, zeroed_z, clipping_distance)

        z_coord = depth_near_center[floor((len(depth_near_center)-1)/2)]

        if x_coord is None:
            continue
        # Append to the file until there is an error at which it will close
        
                # Start the timer


        # Writes the coordinates to each colored object
        csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, color_name + '.csv'))   
        with open(csv_file_path, 'a') as data_to_file:
            data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n') 

        if save_trailing_image:
            if relative_timestamp>5:
                want_trailing = input('Do you want this in trailing image? (y) or (n)')
                if 'y' in want_trailing:
                    #trailing_image_color = trailing_color_temp
                    #cv2.imwrite(data_output_folder_path + '/' + 'trailing' + str(i).zfill(6) +'.jpg',trailing_image_color) 
                    
                    if 'blue' in color_name:
                        color_trail= (135,82,0)
                        radius_trail = 2
                        thickness_trail = 3
                    elif 'green' in color_name:
                        color_trail = (66,132,29)
                        radius_trail = 2
                        thickness_trail = 2
                    else: 
                        color_trail = (255,255,255)
                        radius_trail = 1
                        thickness_trail = 2
                    cv2.circle(trailing_color, (int(x_pixel), int(y_pixel)), radius_trail, color_trail, thickness_trail)
                    cv2.circle(trailing_color, (int(x_pixel), int(y_pixel)), radius_trail+1, (0,0,0), 1)  # Black outline
                    cv2.imwrite(data_output_folder_path + '/' + 'trailing'+ '.jpg',trailing_color) 
                    cv2.imshow('trailing', trailing_color)
                    cv2.moveWindow('trailing',140,140)
        # Always put the circle on the object including the time to correlate the data with the frame
        cv2.circle(cv_color, (int(x_pixel), int(y_pixel)), int(pixel_radius), (0,0,0), 3)
        #cv2.circle(mask, (int(x), int(y)), int(pixelradius), (int(255-(count*255))), 3)
        cv2.circle(depth_colormap, (int(x_pixel), int(y_pixel)), int(pixel_radius), (0,0,0), 3)
         
        
        cv2.putText(cv_color, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

        #print(objectName, x_coord, y_coord, z_coord)
        # Display results
    if x_pixel is None:
        x_coord, y_coord, z_coord = None, None, None
    else:
        if show_image:
            cv2.putText(cv_color, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            cv2.putText(cv_color, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            cv2.putText(cv_color, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            cv2.imshow('color', cv_color)
            cv2.moveWindow('color',0,0)
        if save_image:
            cv2.imwrite(data_output_folder_path + '/' + 'color' + str(i).zfill(6) +'.jpg',cv_color) 
           
            ## TODO pass save_depth_image to this location and have an if statement for depth separate
            cv2.imshow('depth',depth_colormap)
            cv2.moveWindow('depth',500,0)
            cv2.imwrite(data_output_folder_path + '/' + 'depth' + str(i).zfill(6) +'.jpg',depth_colormap)  

    return x_coord, y_coord, z_coord 

    
