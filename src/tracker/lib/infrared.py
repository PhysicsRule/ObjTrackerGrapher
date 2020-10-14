import cv2
import pyrealsense2 as rs
import os
import numpy as np
import argparse

def find_and_config_device(file_path, type_of_tracking):
    pipeline = rs.pipeline()
    config = rs.config()
    if type_of_tracking == 'id':
        config.enable_stream(rs.stream.depth, 848 , 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    else:
        if type_of_tracking == 'all':
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
            config.enable_stream(rs.stream.depth, 848 , 480, rs.format.z16, 60)
            config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 60)
    profile = pipeline.start(config)
    return pipeline

def read_infrared_bounds(): 
# determine the objects you want to track    
 
    num_objects = int(input('How many objects do you want to enter?'))
    for i in range(num_objects):
        object_name = input('what do you want to call the object? Start with THE LARGEST OBJECT AND WORK YOUR WAY DOWN. Examples:  yb or yellowTennisBall')
        print ("Realize the depth camera will measure the object from the front of the object then in the amount you put for the radiusmeters. /n")
        radiusmeters = input("What is the radiusmeters of the object? enter 0.01 if you don't know,")

        mass = input("What is the mass of the object? enter 0.0 if you don't care \n")
        # If we use light and dark color tracking for infrared
        # dt = np.dtype([('lower', np.int32, (3,)),('upper', np.int32, (3,)), ('name', np.unicode_, 16), ('radiusmeters', np.float32),('mass', np.float32)])    
        dt = np.dtype([ ('name', np.unicode_, 16), ('radiusmeters', np.float32),('mass', np.float32)])    

        #standard light object and dark object
        ##  I think these values will for for one light and another dark
        ## TODO This might work but I will most likely abandon
        #dark_min = 0
        #dark_max = 60
        #light_min = 80
        #light_max = 200
        #if i == 0:  # dark object
        #    dark_array = np.array( [((dark_min ,dark_min ,dark_min ),(dark_max, dark_max, dark_max),(objectName),(radiusmeters),(mass))], dtype=dt)
        #else:
        #    light_array = np.array( [((light_min ,light_min ,light_min ),(light_max, light_max, light_max),(objectName),(radiusmeters),(mass))], dtype=dt)
        #    new_monochrome_ranges = np.hstack((dark_array,light_array))          
        #   print(new_monochrome_ranges,' \n')
        
        # If we change our infrared tracking to dark and light use this function
        #thearray = Find_monochrome_bounds(objectName, radiusmeters, mass)

        thearray = np.array( [(object_name,(radiusmeters),(mass))], dtype=dt)
        if i==0:
            new_monochrome_ranges = thearray
        else:
            new_monochrome_ranges = np.hstack((new_monochrome_ranges,thearray))          
                
        print(new_monochrome_ranges,' \n')
        
    # Save array for later
    print ('What do you want to call the scene (name of file) that has each object information? For example dimmOrangeB.')
    name_of_array = input('Just hit enter if you want it to be called, testscene.')

    if name_of_array == '':
        name_of_array = 'testscene'

    basepath = os.getcwd()
    ## TODO send data type to this section
    npy_file = os.path.abspath(os.path.join(basepath, 'data/infrared_i/' + name_of_array))
    np.save(npy_file, new_monochrome_ranges)

    # open last file folder???
    ## Maybe this is a place to use python class???
   
    return i, new_monochrome_ranges
    ## When the program gets more complicated then we can return something different or read from files

def choose_or_create_objects(dir_path_npy, what_to_do_with_npy_files):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-b", "--buffer", type=int, default=64,
	    help="max buffer size")
    ap.add_argument("-r", "--radius", default=10,
        help="minimum radius to be tracked")
    ap.add_argument("-n", "--maxnumpoint", default=2,
        help="maximum number of objects to be plotted in a frame")
    args = vars(ap.parse_args())

    # Set the trackable object
    min_radius_of_object = args["radius"]
    max_num_point = args["maxnumpoint"]
    new_monochrome_ranges = []
    # define the lower and upper boundaries of the several colors
    # in the HSV color space, then initialize the
    # list of tracked points
    
    # enter how you want to get the color_ranges you will use for the boundaries of your objects
    # It will use these boundaries to define.. for example what the red ball looks like
   
    
    if what_to_do_with_npy_files == 'e':
        # erase a *.npy file
        remove_file = input('which file do you want to remove? \n')
        os.path.abspath(os.remove(dir_path_npy + '/' + remove_file))
    
    elif what_to_do_with_npy_files == 'n':
       # Create a new *.npy file
       i, new_monochrome_ranges = read_infrared_bounds()  ## If this doesn't work then:  monochorme_ranges=np.array(new_monochrome_ranges)
       print(new_monochrome_ranges)
       
    elif what_to_do_with_npy_files == 'f':
    # Use an existing *.npy file
        np_file = input("Which file do you want? leave off the extension .npy. If you want testscene just hit enter \n")
        np_file_name = str(np_file + '.npy')
        if np_file =='':
            np_file ='testscene'
        np_file_name = str(np_file + '.npy')
        try:
            npy_file = os.path.abspath(os.path.join(dir_path_npy + '/'+ np_file_name))
            new_monochrome_ranges= np.load(npy_file)
            
            
        except:
            print('We could not find that file. Copy and paste to get spelling correct')
        
    else:
        # If people want the default values d
        new_monochrome_ranges = ("object", 0.0, 1.0)
    
    print(new_monochrome_ranges)
    
    
    return new_monochrome_ranges


def get_depth_without_align(x, y, radiusmeters, rs_depth, rs_infrared, infrared_image, zeroed_x, zeroed_y, zeroed_z, clipping_distance):
    

    # Get depth

    depth = round(rs_depth.get_distance(int(x),int(y)),4) # + radiusmeters  
    #print('x,y,depth depth_pixels. before intrinsics', x,y, depth, '\n')
    if depth - radiusmeters < 0.1 or depth > clipping_distance:
        return None, None, None
    # TODO if this is not a ball then radiusmeters should be 0 as object outline is at the new depth 
    depth = depth + radiusmeters
    
    depth_pixel = [x,y] # object pixel
    
    # TODO check to see if this is really necessary. I do not think it is.
    infrared_intrin = rs_infrared.profile.as_video_stream_profile().intrinsics
    infrared_point = rs.rs2_deproject_pixel_to_point(infrared_intrin, depth_pixel, depth)

    # This is to find the depth using the x,y coordinate in pixels. Then knowing the depth, they can find the x,y in meters
    depth_intrin = rs_depth.profile.as_video_stream_profile().intrinsics
    # The commented code was used when  I used the depthframe. I would like to use it again someday.
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth)

    # Round all 3 coordinates to 5 decimal places
    x_coord = round(depth_point[0], 5)
    y_coord = round(depth_point[1], 5)
    #print('depth points', x_coord,y_coord)

    # The origin in the z axis is the wall in the back you used for clipping distance
    z_coord = 0 - depth   # TODO if reference was the center I already delt with the radius: (zeroed_z-(z+radiusmeters), 5)  depth is front of z_coord

    #print ('cooordinates right after assign', x_coord, y_coord, z_coord)

    x_coord = x_coord - zeroed_x  # Don't flip axis  ## TODO zeroed values are in pixels not meters so convert
    y_coord = round(zeroed_y - y_coord, 5)  # Flip axis so up is+
    #print('zeroed', zeroed_x, zeroed_y)

    return x_coord, y_coord, z_coord

def find_object_by_subtracting_background(i, relative_timestamp, backgd, image, rs_depth, rs_infrared, data_output_folder_path, type_of_tracking, zeroed_x, zeroed_y, zeroed_z, clipping_distance, new_monochrome_ranges, min_radius_of_object, max_num_point):
    # construct a mask by subtracting the new infrared image from the background infrared image, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    
    # creates an infrared frame of just the moving object
    # It is a mask the difference between the background and the current frame
    #cv2.imshow('image', image)
    #cv2.imshow('backd', backgd)
    frame_delta = cv2.absdiff(backgd, image)
    max_value = np.max(frame_delta)
    #cv2.imshow('framedelta', frame_delta)
    #nothingImportant = cv2.selectROI('frameDelta', frame_delta, False)

    ## TODO Have user be able to modify threshold values
    # lower the # the more sensitive it is a tracking something with with similar color to backgorund
    thresh = cv2.threshold(frame_delta, 8
    , max_value, cv2.THRESH_BINARY)[1]  
    # used to be thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]  

    #nothingImportant = cv2.selectROI('threshold', thresh, False)
    # cv2.imshow('framedelta', frame_delta)

    #stacked = np.hstack((infrared_backgd,infrared_image,frameDelta))
    # Show this stacked frame at 40% of the size.
    #cv2.imshow('background-image-delta',cv2.resize(stacked,None,fx=0.4,fy=0.4))

    # mask = cv2.inRange(fgmask, lower,upper)  F use something similar to detect light or dark object moving
    mask = thresh

    ## TODO clip the parts of the image that are in the background 
    mask_in_Range = mask

    
    #if type_of_tracking == 'back':
    #    mask = cv2.dilate(mask, None, iterations=4)

    #cv2.imshow('mask before erode', mask)
    mask = cv2.erode(mask, None, iterations=2)
    cv2.imshow('mask after erode', mask)

    if type_of_tracking == 'backd':
        cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
        mask = mask.astype(np.uint8)
    # Convert the depth image to an 8 bit imageby normalizing it between 0 and 255
    
    #cv2.imshow('mask before erode2', mask)
    #mask = cv2.erode(mask, None, iterations=2)
    #cv2.imshow('mask after erode2', mask)
    # find contours in the mask and initialize the current
    
    # Extract contours from image
    # https://stackoverflow.com/questions/61450506/how-do-i-contour-multiple-largest-objects-in-an-image
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print('contours out of loop', contours)
    #contoursSorted = contour_areas.argsort()
    contour_areas = np.array([cv2.contourArea(contour) for contour in contours])

    
    #print(len(contoursSorted), 'length of contour')
    count = 0    
    x = None
    x_coord =  None
    # Inspect contours if they exist
    # returns 1 or two contours
    # TODO organize contours as larger and smaller. Right now it is top and bottom contour
    check_get_point = False

    # SET to True if you want to show the ball getting tracked. Note: This will reduce your framerate
    show_image = True

    # SET to True if you want all of the infrared images saved
    save_image = True
    while (count+1)<= max_num_point and contours and len(contours)>= (count+1):
        #print(' length of contours', len(contours))
        check_get_point = False
        # Find the largest contour, then delete it so the next round will have the smaller one
        #print('contours in loop', contours)
        #print(np.shape(contours), '\n area', contour_areas, '\n which is max#', np.argmax(contour_areas))
        #print('largest contour to keep', int(np.argmax(contour_areas)))
        largest_contour = contours[int(np.argmax(contour_areas)):][0]

        if len(contour_areas)>1:
            contour_areas[ np.argmax(contour_areas)] = 0
        #max_index = contours_sorted[::(-1-count)][0]
        #print(max_index)
        #largest_contour = contours[max_index] #largest contour then the next largest etc.
        
        ((x, y), pixelradius) = cv2.minEnclosingCircle(largest_contour)
        #print('largest contour',largest_contour)
        #print(x,y, pixelradius, ' count', count)
        if pixelradius > min_radius_of_object:
            cv2.circle(image, (int(x), int(y)), int(pixelradius), (255-(count*255)), 2)
            object_name, radiusmeters, mass = new_monochrome_ranges[count]
            #print(255-(count*255), 'color of object', object_name)

            if x is None:
                count += 1
                #print('x is none')
                continue
            
            check_get_point = True
            x_coord, y_coord, z_coord = get_depth_without_align(x,y, radiusmeters, rs_depth, rs_infrared, image, zeroed_x, zeroed_y, zeroed_z, clipping_distance)
            
            #print (count, object_name, x_coord,y_coord,z_coord, 'radius meters', radiusmeters)

            # Converts time from milliseconds to seconds
            # Append to the file until there is an error at which it will close
            
            if x_coord is None:
                count +=1
                continue

            # Writes the coordinates to each colored object
            csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, object_name + '.csv'))   
            with open(csv_file_path, 'a') as data_to_file:
                data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n') 
                

            # Always put the circle on the object including the time to correlate the data with the frame
            cv2.circle(image, (int(x), int(y)), int(pixelradius), (int(255-(count*255))), 3)
            #cv2.circle(mask, (int(x), int(y)), int(pixelradius), (int(255-(count*255))), 3)

            cv2.putText(image, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

        #print(objectName, x_coord, y_coord, z_coord)
        # Display results
        if show_image:
            cv2.putText(image, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            cv2.putText(image, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            cv2.putText(image, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2) 

        count +=1
        #print(count) 

        # exit if spacebar or esc is pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == 32:
            print('end')
            break
    
    if show_image:
        cv2.imshow('infrared', image)
        cv2.moveWindow('infrared',0,0) # 848 , 480 
        #cv2.imshow('mask', mask)
        #cv2.moveWindow('mask',500,240) # 848 , 480 
        
    if save_image:
        cv2.imwrite(data_output_folder_path + '/' + 'infrared' + str(i).zfill(6) +'.jpg',image) 
             

    if x is None:
        x_coord, y_coord, z_coord = None, None, None  
    if check_get_point == False:
        return None, None, None
    else: 
        return x_coord, y_coord, z_coord

def find_object_by_tracker(bbox, i, relative_timestamp, backgd, image, rs_depth, rs_infrared, data_output_folder_path, type_of_tracking, zeroed_x, zeroed_y, zeroed_z, clipping_distance, new_monochrome_ranges, max_num_point):
    def draw_bounding_box(image, bounding_box):
    # Draws a box to indicate what object is being tracked and noting its starting position.
        x_coordinate = int(bounding_box[0])
        y_coordinate = int(bounding_box[1])
        width = int(bounding_box[2])
        height = int(bounding_box[3])
        cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (255, 0,0), 3,1)

    save_image = True
    show_image = True
    check_get_point = True
    # TODO add multiple objects and have a count of 0 and 1 so 2 objects
    count=0
    object_name, radius_meters, mass = new_monochrome_ranges[count]
    draw_bounding_box(image, bbox)
    x = int(bbox[0]+bbox[2]/2)
    y = int(bbox[1]+bbox[3]/2)
    
    x_coord, y_coord, z_coord = get_depth_without_align(x, y, radius_meters, rs_depth, rs_infrared, image, zeroed_x, zeroed_y, zeroed_z, clipping_distance)
    print(x, y, radius_meters, x_coord)
     # Writes the coordinates to each colored object
    csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, object_name + '.csv'))   
    with open(csv_file_path, 'a') as data_to_file:
        data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n') 
        

    cv2.putText(image, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    #print(objectName, x_coord, y_coord, z_coord)
    # Display results
    #if show_image:
    cv2.putText(image, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv2.putText(image, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv2.putText(image, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)     
    if show_image:
        cv2.imshow('infrared', image)
        cv2.moveWindow('infrared',0,0) # 848 , 480 
        #cv2.imshow('mask', mask)
        #cv2.moveWindow('mask',500,240) # 848 , 480 
        
    if save_image:
        cv2.imwrite(data_output_folder_path + '/' + 'infrared' + str(i).zfill(6) +'.jpg',image) 
             

    if x is None:
        check_get_point = False
    if check_get_point == False:
        return None, None, None
    else: 
        return x_coord, y_coord, z_coord
    return x_coord, y_coord, z_coord