# Takes a *.csv file named *.csv with the headers: Time, x,y,z
# 9 Graphs are the result 
## Position vs. time, velocity vs. time, and either momentum or acceleration vs. time (along all 3 coordinate axes)
## Energy vs. time will be included again in the future
# The program will find a trendline for x,y,z given a domain (tmin and tmax) to create other graphs in the future.
# For now, use graphing_trendlines.exe to get the trendliens until the GUI version has been added to the new platform.
### Users state either linear or quadratic fits.
###                exp and sine graphs don't work

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pandas as pd
from scipy import *
import os
import time

import numpy as np

from tracker.lib.general import GUI_creates_an_array_of_csv_files
from tracker.lib.setup_files import create_new_folder, create_calc_file
from tracker.lib.user_input import select_type_of_tracking, select_multiple_files
from tracker.lib.graphing import plot_graphs, GUI_trim, trim_from_collision, best_fit_fun_graph, three_D_graphs, FindVelandAccTrend, FindMom, FindKE, FindTotalEnergy
from mpl_toolkits import mplot3d

def plot_style_color():
    show_legend =  True
    line_style_array = ("solid", "dashed", "dotted", "dashdot")
    line_color_array = ("red", "blue", "green", "cyan", "magenta")
    marker_shape_array = ("^","s","o","*","X")

    # Setup the graph and determine the filepaths for the files
    print("We can graph up to 5 objects unless you want to add to the line styles")
    print(
        "The graphs will be color coded, if doing colored object select them in the following order:"
    )
    print(line_color_array)
    print(line_style_array)
    return line_style_array, line_color_array, marker_shape_array, show_legend

#TODO input_folder has the radius and mass of the object so it can be used later for momentum
def GUI_graph (which_parameter_to_plot, data_output_folder_path, graph_color_ranges, csv_files_array, points_to_smooth ):

    # Setup Trendline folder. I
    # If user looks at a different variable or time domain, new folder created.
    for (file_path, file_name, mass) in csv_files_array:
        print(file_path, file_name)
    new_sheet_folder = 'trendlines'
    trendline_folder_path = create_new_folder (file_path, new_sheet_folder)
    #create a file to store the trendline equations 
    calc_file_name =  "calcs.csv"
    calc_file_name_path = os.path.abspath(os.path.join(trendline_folder_path + '/' + calc_file_name + '/' ))   
    create_calc_file(calc_file_name_path)

    sum_mass = 0

    plt.ion()

    # If there is a collision, the program will track the ball.
    # t1 is designated for the actual throw if used later
    # # before the collision t2-t3
    # after the collision t4-45



    object_number = 0
    for (file_path, file_name, mass) in csv_files_array:
        print(file_name)
        # Write the name to the calculations csv file where the trendlines are

        if which_parameter_to_plot != "a":
            sum_mass = sum_mass + mass
            if which_parameter_to_plot == "E":
                vert_axis = input("Which axis is up (x,y, or z)?")
                zero_ref_for_PE = float(
                    input(
                        "What value in meters do you want your zero reference frame. Note: you might want to say zero on the first round then modify."
                    )
                )
        object_number += 1
    
    # When smoothing data for finding slopes and approximate derivatives (can also use 5, 7, etc.)
    # The more points, the less distinct collisions and more inaccurate the first few and last few points of slope data
    PointsToSmooth = 3
    # Column header on position data 
    header_list = ['Time', 'x', 'y', 'z']
    # i represents each object tracked
    i=0
    object_number = 0
    for (file_path, file_name, mass) in csv_files_array:
    # For each object create a data_file from position data

        # Setup data_frame to put velocity, momentum, acceleration, and energy data
        file_name_dataframe = file_name  + "sheet.csv"
        file_name_dataframe_path = os.path.abspath(os.path.join(trendline_folder_path + '/' + file_name_dataframe + '/' ))   

        # Reads the current version of the *csv file and smooths it
        file_name_w_extension = file_name + '.csv'
        path_to_file = os.path.abspath(os.path.join(file_path, file_name_w_extension))
        graph_data = pd.read_csv(path_to_file, header=0, names = header_list)
        
        data_frame = pd.DataFrame(graph_data) 
        smooth_data_to_graph = data_frame.set_index('Time')

        smooth_data_to_graph.to_csv(file_name_dataframe_path) 
        print('Your folder will be available:\n', file_name_dataframe_path)    
        smooth_data_to_graph = pd.read_csv(file_name_dataframe_path, header=0)

        # Using weighted differences find the velocity and acceleration graphs
        __, graph_data, smooth_data_to_graph = FindVelandAccTrend(
            smooth_data_to_graph, points_to_smooth
        )
        # Use the velocity data to find momentum or energy
        if which_parameter_to_plot == "p":
            __, graph_data, smooth_data_to_graph = FindMom(smooth_data_to_graph,
                smooth_data_to_graph, header_list, points_to_smooth, mass
            )
        if which_parameter_to_plot == which_parameter_to_plot == "E":
            __, graph_data, smooth_data_to_graph = FindKE(smooth_data_to_graph,
                data_frame, header_list, points_to_smooth, mass
            )
            __, graph_data, smooth_data_to_graph = FindTotalEnergy(smooth_data_to_graph, 
                header_list, points_to_smooth, mass, vert_axis, zero_ref_for_PE
            )

        smooth_data_to_graph.to_csv(file_name_dataframe_path) 

        object_number += 1

 
        i += 1

    # Allow the input screen instead of the graph
    plt.ioff()


    # i is the object number    
    i = 0
    sum_mass = 0
    return trendline_folder_path, smooth_data_to_graph


def GUI_graph_trendline (fig, axes, line_style_array, line_color_array, which_parameter_to_plot, showlegend, trendline_folder_path, graph_color_ranges, csv_files_array ):
## TODO The values for the trendlines times will be modified through the GUI in the future
    i = 0
    collision = 'n'
    ##collision = input('Do you want trendlines from a collision?\n')
    if collision in 'yes':
        trendline_times = [None]*6
        trendline_times[0] = 0
        trendline_times[1] = 0
        trendline_times[2] = 0.2
        trendline_times[3] = 0.4
        trendline_times[4] = 0.6
        trendline_times[5] = 0.8
        print(trendline_times)
    
    xmin = 0.0
    xmax= 0.6

    calc_file_name =  "calcs.csv"
    calc_file_name_path = os.path.abspath(os.path.join(trendline_folder_path + '/' + calc_file_name + '/' ))   
    create_calc_file(calc_file_name_path)
    object_number = 0
    for (file_path, file_name, mass) in csv_files_array:  
        # find the domain for each trendline, create a new *.csv for each trendline, 

        s = 0

        # The first trendline starts at time of t2
        trendline_time_counter = 2
        print('t', trendline_time_counter)
        
        plt.ioff()
        
        while True:
            with open(calc_file_name_path, 'a') as calcs_to_file:
                calcs_to_file.write(f'{file_name}\n') 
            print(file_name)
        # Let the user try multiple trendline for each object.They must do one for each object
            file_name_dataframe = file_name  + "sheet.csv"
            file_name_dataframe_path = os.path.abspath(os.path.join(trendline_folder_path + '/' + file_name_dataframe + '/' ))   
            
            graph_dataZoomed = pd.read_csv(file_name_dataframe_path, header=0)

            # Find minimum and maximum for trendlines
            if 'y' in collision:
                graph_data_window = trim_from_collision(graph_dataZoomed, calc_file_name_path, trendline_times, trendline_time_counter)
        
            else: 
                graph_data_window = GUI_trim(graph_dataZoomed, calc_file_name_path, xmin, xmax)

            # PFind trendlines and graph
            graph_data_window = best_fit_fun_graph(fig, axes, graph_data_window, line_style_array[i], line_color_array[i], which_parameter_to_plot, mass,file_name_dataframe_path, calc_file_name_path)
            
            file_name_dataframe_trendline = file_name  + str(s) + "trend.csv"
            file_name_dataframe_path_trendline = os.path.abspath(os.path.join(trendline_folder_path + '/' + file_name_dataframe_trendline + '/' ))   
            graph_data_window.to_csv(file_name_dataframe_path_trendline) 

            time.sleep(1)
            
            # Turns the plot back on and displays the curve fits
            # If the user wants another trendline do the loop again, otherwise stop the program
            
            s += 1        
            
            if 'y' in collision:
            # If the trendlines are being calculated automatically from the initial collision times
                if trendline_time_counter <3:
                    trendline_time_counter +=2 # Now it is 4 (after the collision)
                else:
                    break
            else: 
            # User is manually putting in the trendlines as they go for any situation
                trendline_check = 'n'    
                if trendline_check != 'y':
                    break

        object_number += 1      
        i += 1
        plt.tight_layout()
        plt.ioff()
        
        
    # exit if spacebar or esc is pressed
    #input('press return to finish')   

    # Save the image of the graphs when you close it
    image_name= "graph_" + which_parameter_to_plot + ".png"
    graph_path = os.path.abspath(os.path.join(trendline_folder_path, image_name ))   
    fig.savefig(graph_path)
    plt.show()  
    plt.ioff()

            
