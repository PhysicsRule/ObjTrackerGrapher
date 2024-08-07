# Parts of the program that specifically use the Intel RealSense D435i, 
# but with the new changes that has not been verified.
# In the future there may be other 3D cameras that can be chosen instead.


from typing import Any, Tuple, Optional
import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os

def warm_up_camera(pipeline) -> None:
    # Read video frame after waiting a bit for the camera to warm up
    # Create a pipeline object. This object configures the streaming camera and owns it's handle

    i=1
    check = True
    while i<20 and check:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
        i += 1
        # Attempted Queue
        # frames = pipeline.wait_for_frame()
        frames = pipeline.wait_for_frames()
        if not frames:
            check = False

def find_and_config_device():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
    profile = pipeline.start(config)
    warm_up_camera(pipeline)
    # Let the camera run for a few seconds so you do not get dark images
    
    return pipeline

def find_and_config_device_mult_stream(types_of_streams_saved) -> Any:
    pipeline = rs.pipeline()
    
    config = rs.config()
    if types_of_streams_saved == 'id':
        config.enable_stream(rs.stream.depth, 848 , 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    elif types_of_streams_saved == 'id300':
            config.enable_stream(rs.stream.depth, 848 , 100, rs.format.z16, 300)
            config.enable_stream(rs.stream.infrared, 1, 848, 100, rs.format.y8, 300)
    else: # (cd)color
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        if types_of_streams_saved == 'all':
            config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 60)
    
    # If you want to use multiple cameras in the future this is useful
    #ctx = rs.context()
    #if len(ctx.devices) > 0:
    #    for d in ctx.devices:
    #        SerialForD435i = d.get_info(rs.camera_info.serial_number)
    #        print ('Found device: ', \
    #            d.get_info(rs.camera_info.name), ' ', \
    #            SerialForD435i)
    #else:
    #    print("No Intel Device connected")
    #config.enable_device(SerialForD435i)

    # Streams will get recorded to this .bag file
    '''
    sensors = profile.get_device().query_sensors()
    for sensor in sensors:
        if sensor.supports(rs.option.auto_exposure_priority):
			#print('Start setting AE priority.')
            aep = sensor.get_option(rs.option.auto_exposure_priority)
            print(str(sensor),' supports AEP.')
            print('Original AEP = %d' %aep)
            aep = sensor.set_option(rs.option.auto_exposure_priority, 0)
            aep = sensor.get_option(rs.option.auto_exposure_priority)
            print('New AEP = %d' %aep)
            #ep = sensor.set_option(rs.option.exposure, 78)
            #ep = sensor.set_option(rs.option.enable_auto_exposure)
    '''

    return pipeline, config


def record_bag_file(data_output_folder_path, types_of_streams_saved):
# Record Images from pipeline
# DOESN't Work as it cannot open the bag file for some reason
# For now, just use the Intel Real Sense View if you installed it
# https://www.intelrealsense.com/sdk-2/
    filepath_bag = os.path.abspath(os.path.join(data_output_folder_path, 'bag.bag'))
    pipeline, config = find_and_config_device_mult_stream(types_of_streams_saved)
    config.enable_record_to_file(filepath_bag)
    pipeline.start(config)
    warm_up_camera(pipeline)
    time.sleep(1)
    #wait_to_start = input('Hit enter to start and stop recording')
    print('still recording')
    
    time_to_record = 6 # seconds
    if types_of_streams_saved == 'cd': frames_to_record = time_to_record * 60
    elif types_of_streams_saved == 'id300': frames_to_record = time_to_record * 300
    elif types_of_streams_saved == 'id': frames_to_record = time_to_record * 90

    # TODO Change this to while true and break at space bar when done testing
    for _ in range(frames_to_record):
        frames = pipeline.wait_for_frames()
        # k = cv2.waitKey(1) & 0xff
        # if k == 27 or k == 32:
        #    print('end')
        #    break
    pipeline.stop()
    # read_bag_file_and_config(types_of_streams_saved, data_output_folder_path, 'bag', filepath_bag)
    print('done recording')

def get_all_frames_color(rs_pipeline) -> Optional[Tuple[Tuple[Any, Any, Any], Any]]:
    '''
    Returns a tuple containing the OpenCV color and (aligned) RealSense
    depth/color frames, as well as the timestamp of the frames. If either frame
    doesn't exist, returns `False` instead.
    '''  
    # Get & align RealSense frames
    rs_frames = rs_pipeline.wait_for_frames()
    timestamp = rs_frames.get_timestamp()

    # Get OpenCV frame (or handle if it isn't read properly)

    rs_align = rs.align(rs.stream.color)
    rs_frames_aligned = rs_align.process(rs_frames)

    # Extract color/depth frames
    rs_color = rs_frames_aligned.get_color_frame()
    rs_depth = rs_frames_aligned.get_depth_frame().as_depth_frame()
    cv_color = np.array(rs_color.get_data())

    # Check that both frames are valid & return false if they aren't
    if not rs_depth or not rs_color:
        return None

    # Calculate elapsed time from start_datetime, if applicable  
    return (cv_color, rs_color, rs_depth), timestamp

def get_all_frames_infrared(rs_pipeline) -> Optional[Tuple[Tuple[Any, Any, Any], Any]]:
    '''
    Returns a tuple containing the OpenCV color and (aligned) RealSense
    depth/color frames, as well as the timestamp of the frames. If either frame
    doesn't exist, returns `False` instead.
    '''  
    # Get & align RealSense frames
    rs_frames = rs_pipeline.wait_for_frames()
    timestamp = rs_frames.get_timestamp()

    # Extract color/depth frames
    rs_infrared = rs_frames.get_infrared_frame()
    rs_depth = rs_frames.get_depth_frame().as_depth_frame()

    # Check that both frames are valid & return false if they aren't
    if not rs_depth:
        return None

    # Calculate elapsed time from start_datetime, if applicable  
    return (rs_depth, rs_infrared), timestamp


def select_clipping_distance(frame, rs_depth) -> Tuple[Any, float, float]:
# This selects both the background and the TODO  Origin
     # Find distance (depth) to wall & store as clipping distance
    # Note that it is reduced slightly to allow some room for error (*0.95)
    print('Select a box at a distance further than points will be collected.')
    print('Any colored object beyond that point will not be tracked, but may slow down the data collection')
    wall = cv2.selectROI('Select an object/wall behind the movement to remove unwanted data. Then ENTER. Then SPACEBAR', frame, False)
    depth = round(rs_depth.get_distance(wall[0], wall[1]), 4)
    clipping_distance =  depth  * 0.99
    return wall, depth, clipping_distance

def get_coordinates_meters(rs_frame, rs_depth, x_pixel: int, y_pixel: int, depth) -> Tuple[float, float, float]:
# Given the x,y in pixels find the x,y,z in meters    
    
    depth_pixel = [int(x_pixel),int(y_pixel)] # object pixel
    depth_intrin = rs_depth.profile.as_video_stream_profile().intrinsics        
    depth_to_frame_extrin = rs_depth.profile.get_extrinsics_to(rs_frame.profile)
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth)

    # frame_point is from the perspective of the color or infrared camera whichever was used
    frame_point = rs.rs2_transform_point_to_point(depth_to_frame_extrin, depth_point)
    return frame_point

def select_location(frame, rs_frame, rs_depth) -> Tuple[float, Tuple[float, float, float], float]:
# This selects both the background and the TODO  Origin
     # Find distance (depth) to wall & store as clipping distance
    # Note that it is reduced slightly to allow some room for error (*0.95)
    wall, depth, clipping_distance = select_clipping_distance(frame, rs_depth)
    depth = round(rs_depth.get_distance(int(wall[0]),int(wall[1])),4) 
    frame_point = get_coordinates_meters(rs_frame, rs_depth, wall[0], wall[1], depth)
    return clipping_distance, frame_point, depth

def get_depth_meters(x_pixel, y_pixel, radius_meters, rs_depth, rs_frame, zeroed_x, zeroed_y, zeroed_z, clipping_distance) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        depth = round(rs_depth.get_distance(int(x_pixel),int(y_pixel)),4) 
        if (depth - radius_meters) < 0.1:
            print('too close', depth)
            return -1, -1, -1
        elif depth > clipping_distance:
            print('too far  ', depth)
            return -1, -1, -1
    except:
        print('error in getting depth')
        return -1, -1, -1
    
    # Given the location in pixels for x,y and the depth, find the coordinates in meters
    frame_point = get_coordinates_meters(rs_frame, rs_depth, x_pixel,  y_pixel, depth)
    
    # Round all 3 coordinates to 5 decimal places
    x_coord = round(frame_point[0], 5)
    y_coord = round(frame_point[1], 5)

    x_coord = x_coord - zeroed_x  # Don't flip axis
    y_coord = round(zeroed_y - y_coord, 5)  # Flip axis so up is+
    
    # The origin in the z axis is the wall in the back you used for clipping distance
    # TODO if you want to use the zeroed_z as the reference point replace 0 with zeroed_z: (zeroed_z-(z+radiusmeters), 5)  
    z_coord =round(0 - (depth + radius_meters) , 5)  
    return x_coord, y_coord, z_coord


def set_the_origin(point) -> Tuple[float, float, float]:
    # Use the furthest point as the frame of reference, but it is overwritten for now
    # Round all 3 coordinates to 5 decimal places
    zeroed_x = round(point[0], 5)
    zeroed_y = round(point[1], 5)
    # let the user set the reference point if they want to once the config file is set up
    ## TODO 
    zeroed_x = 0 # will need to be in meters
    zeroed_y = 0 # for now
    zeroed_z = 0
    
    return zeroed_x, zeroed_y, zeroed_z

def select_furthest_distance_color(pipeline) -> Tuple[float, float, float, float]:
    # Selects the furthest distance and sets the origin to be the top left corner
    ## TODO add another selection box to say what the x,y,z origin should be after we are confident of z axis
    # OpenCV color frame
    cv_color = None
    # Select background as a maximum depth
    clipping_distance = 0

    while clipping_distance < 0.1:
        # Get the OpenCV color/RealSense depth frames & skip iteration if
        # necessary
        frame_result = get_all_frames_color(pipeline)
        if not frame_result:
            continue
        (cv_color, rs_color, rs_depth), _ = frame_result
        # Find the clipping distance
        clipping_distance, color_point, zeroed_z = select_location(cv_color, rs_color, rs_depth)
        print('clipping distance is', clipping_distance)
        #TODO later call above to find the origin then call the following after
        # for now this just sets all zeroed values as 0 insted of a specific origin, so the origin is the cent of the camera
        zeroed_x, zeroed_y, zeroed_z = set_the_origin (color_point)

    print('Hit space bar to start and stop recording.')
    space = 0
    while space != 32:
        space = cv2.waitKey(1) & 0xff
    # Remove ROI selection window, as it is no longer necessary
    cv2.destroyWindow('Select a clipping distance. Is will be the upper left of box')
    #cv2.destroyWindow('ROI Selection')

    return zeroed_x, zeroed_y, zeroed_z, clipping_distance 


def select_furthest_distance_infrared(pipeline):
    # Selects the furthest distance and sets the origin to be the top left corner
    ## TODO add another selection box to say what the x,y,z origin should be after we are confident of z axis
    # OpenCV color frame
    
    # Select background as a maximum depth
    clipping_distance = 0
    while clipping_distance < 0.1:
        # Get the OpenCV color/RealSense depth frames & skip iteration if
        # necessary
        frame_result = get_all_frames_infrared(pipeline)
        if not frame_result:
            continue
        (rs_depth, rs_infrared), _ = frame_result
        if not rs_infrared:
                continue
        infrared_image = np.asanyarray(rs_infrared.get_data())
        # Find distance (depth) to wall & store as clipping distance
        # Note that it is reduced slightly to allow some room for error (*0.95)
         # Find the clipping distance
        clipping_distance, depth_point, zeroed_z = select_location(infrared_image, rs_infrared, rs_depth)
        #TODO later call above to find the origin then call the following after
        # for now this just sets all zeroed values as 0 instead of a specific origin, so the origin is the cent of the camera
        zeroed_x, zeroed_y, zeroed_z = set_the_origin (depth_point)

    ## TODO add this back in if we use background subtraction again
    '''print('Hit space bar to start and stop recording')
    space = 0
    while space != 32:
        space = cv2.waitKey(1) & 0xff
    '''
    # Remove ROI selection window, as it is no longer necessary
    cv2.destroyWindow('ROI Selection')

    return zeroed_x, zeroed_y, zeroed_z, clipping_distance


def read_bag_file_and_config(types_of_streams_saved, data_output_folder_path, folder_name , bag_folder_path) -> Any:
    try:
        config = rs.config()
        #bag_path =  os.path.abspath(os.path.join(bag_folder_path + "/" + bag_file ))
        rs.config.disable_all_streams(config)
        rs.config.enable_device_from_file(config, bag_folder_path, repeat_playback=False)
        pipeline = rs.pipeline()        
        if types_of_streams_saved == 'id':
            config.enable_stream(rs.stream.depth)       #, 848, 480, rs.format.z16, 90)
            config.enable_stream(rs.stream.infrared)    #, 1, 848, 480, rs.format.y8, 90)
        elif types_of_streams_saved == 'id300':
            config.enable_stream(rs.stream.depth)       #, 848, 480, rs.format.z16, 300)
            config.enable_stream(rs.stream.infrared)    #, 1, 848, 480, rs.format.y8, 300)
        else:
            ## TODO increase to 60 fps and set S2_OPTION_AUTO_EXPOSURE_PRIORITY to 0 to maintain constant fps when recording
            config.enable_stream(rs.stream.depth)       #, 848, 480, rs.format.z16, 60)
            config.enable_stream(rs.stream.color)       #, 848, 480, rs.format.bgr8, 60)
            if types_of_streams_saved == 'all':
                config.enable_stream(rs.stream.infrared) #, 1, 848, 480, rs.format.y8, 60)

        #pipeline.start(config)
        profile = pipeline.start(config)
        print('reading bag file')
        time.sleep(1)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        time.sleep(2)

        '''
        i = 0
        while True:
            print("Saving frame:", i)
            # Check if new frame is ready
            rs_frames = pipeline.poll_for_frames()
            rs_align = rs.align(rs.stream.color)
            rs_frames_aligned = rs_align.process(rs_frames)

            # Extract color/depth frames
            rs_color = rs_frames_aligned.get_color_frame()
            rs_depth = rs_frames_aligned.get_depth_frame().as_depth_frame()
            depth_image = np.asanyarray(rs_depth.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)
            color_image = np.asanyarray(rs_color.get_data())

            cv2.imwrite(data_output_folder_path + "/" + "depth" + str(i).zfill(6) + ".png", depth_colormap)
            cv2.imwrite(data_output_folder_path + "/" + type_of_tracking + str(i).zfill(6) + ".png", color_image)
            i += 1
            
            rs_frames = rs.composite_frame(rs.frame())
            if pipeline.poll_for_frames(rs_frames):
                rs_align = rs.align(rs.stream.color)
                rs_frames_aligned = rs_align.process(rs_frames)

                # Extract color/depth frames
                rs_color = rs_frames_aligned.get_color_frame()
                rs_depth = rs_frames_aligned.get_depth_frame().as_depth_frame()
                depth_image = np.asanyarray(rs_depth.get_data())
                color_image = np.asannyarray(rs_color.get_data())
                cv2.imwrite(data_output_folder_path + "/" + "depth" + str(i).zfill(6) + ".png", depth_image)
                cv2.imwrite(data_output_folder_path + "/" + type_of_tracking + str(i).zfill(6) + ".png", color_image)
            
                i += 1
            # wait logic until frame is ready
            else:
                print("Waiting for frame to be ready")
            '''
        
    except RuntimeError:
        print("There are no more frames left in the .bag file!")
    finally:
        pass
    return pipeline
