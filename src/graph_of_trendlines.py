# Takes a *.csv file named *.csv with the headers: Time, x,y,z
# The program finds a trendline for x,y,z given a domain (tmin and tmax) to create other graphs
# 9 Graphs are the result. 


### NOTES TO SELF: make the file and trendline an input again when done debugging and remark trim()
###                exp and sine graphs don't work
###                trim off time and xmin xmax the plot
###                use poly for linear, quadratic, etc.. possibly taylor series...



import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pandas as pd
from scipy import *
import os
import time

import numpy as np

from tracker.lib.setup_files import create_new_folder, create_calc_file
from tracker.lib.user_input import select_type_of_tracking, select_multiple_files
from tracker.lib.graphing import graph_setup, plot_graphs, trim, trim_from_collision, best_fit_fun_graph, three_D_graphs, FindVelandAccTrend, FindMom, FindKE, FindTotalEnergy
from mpl_toolkits import mplot3d

#### MAIN PROGRAM ####
# Setup graph legend

#TODO input_folder has the radius and mass of the object so it can be used later for momentum
type_of_tracking, _, data_output = select_type_of_tracking() 
file_type = '.csv'

num_files, file_array = select_multiple_files (data_output, file_type)

# Setup Trendline folder. I
# If user looks at a different variable or time domain, new folder created.
for (file_path, file_name) in file_array:
    print(file_path, file_name)
new_sheet_folder = 'trendlines'
data_output_folder_path = create_new_folder (file_path, new_sheet_folder)
#create a file to store the trendline equations 
calc_file_name =  "calcs.csv"
calc_file_name_path = os.path.abspath(os.path.join(data_output_folder_path + '/' + calc_file_name + '/' ))   
create_calc_file(calc_file_name_path)

mass = [None]*len(file_array)
sum_mass = 0

plt.ion()
line_style_array = ("solid", "dashed", "dotted", "dashdot", (0,(1,10)), ((0,(1,1))), (0,(5,1)), (0,(3,10,1,10)), (0,(3,5,1,5)), (0,(3,10,1,10,1,10))) 
line_color_array = ("red", "blue", "green", "cyan", "magenta", "orange", "purple", "brown", "cyan", "olive" )
marker_shape_array = ("^","s","o","*","X", ".", "1","2","4","8","s")


# Setup the graph and determine the filepaths for the files
print("We can graph up to 5 objects unless you want to add to the line styles")
print(
    "The graphs will be color coded, if doing colored object select them in the following order:"
)
print(line_color_array)
print(line_style_array)


which_parameter_to_plot = input(
    "We will plot position, velocity and one other graph. What is your third graph? (a) acceleration, (p) momentum, (E)total energy?"
)
showlegend = input("Do you want to display a legend? 'y' or 'n' ")

# If there is a collision, the program will track the ball.
# t1 is designated for the actual throw if used later
# # before the collision t2-t3
# after the collision t4-45

collision = input('Do you want trendlines from a collision?\n')
if collision in 'yes':
    trendline_times = [None]*6
    trendline_times[0] = 0
    trendline_times[1] = float(input('What is the time the object starts to move? t1='))
    trendline_times[2] = float(input('What is the time person releases the object completely? t2='))
    trendline_times[3] = float(input('What is the time just before collision? t3='))
    trendline_times[4] = float(input('What is the time just after the collision? t4='))
    trendline_times[5] = float(input('What is the last Time both objects are well tracked (very little blur)? t5='))
    print(trendline_times)

object_number = 0
for (file_path, file_name) in file_array:
    print(file_name)
    # Write the name to the calculations csv file where the trendlines are

    if which_parameter_to_plot != "a":
        mass[object_number] = float(input("What is the mass of your object?"))
        sum_mass = sum_mass + mass[object_number]
        if which_parameter_to_plot == "E":
            vert_axis = input("Which axis is up (x,y, or z)?")
            zero_ref_for_PE = float(
                input(
                    "What value in meters do you want your zero reference frame. Note: you might want to say zero on the first round then modify."
                )
            )
    object_number += 1
fig, axes, points_to_smooth, header_list, fig_3D, axes_3D = graph_setup(which_parameter_to_plot)

# Instruction on how to use the graph
## It would be nice if this was showing on the screen with the graphs
print("When you can see your data, zoom into the section you want to see.")
print("Then hit backarrow to see it for a longer time.")
print("Figure out what time interval you will want to see")
print("Close the graph window to save the file")

# i represents each object tracked
i=0
object_number = 0
for (file_path, file_name) in file_array:
# For each object create a data_file from position data

    # Setup data_frame to put velocity, momentum, acceleration, and energy data
    file_name_dataframe = file_name  + "sheet.csv"
    file_name_dataframe_path = os.path.abspath(os.path.join(data_output_folder_path + '/' + file_name_dataframe + '/' ))   

    # Reads the current version of the *csv file and smooths it
    path_to_file = os.path.abspath(os.path.join(file_path, file_name))
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
            smooth_data_to_graph, header_list, points_to_smooth, mass[object_number]
        )
    if which_parameter_to_plot == which_parameter_to_plot == "E":
        __, graph_data, smooth_data_to_graph = FindKE(smooth_data_to_graph,
            data_frame, header_list, points_to_smooth, mass[object_number]
        )
        __, graph_data, smooth_data_to_graph = FindTotalEnergy(smooth_data_to_graph, 
            header_list, points_to_smooth, mass[object_number], vert_axis, zero_ref_for_PE
        )

    smooth_data_to_graph.to_csv(file_name_dataframe_path) 

    object_number += 1

    three_D_graphs(axes_3D, smooth_data_to_graph,line_style_array[i], line_color_array[i],marker_shape_array[i])
 
    plot_graphs(smooth_data_to_graph, line_style_array[i],line_color_array[i], marker_shape_array[i], fig, axes, which_parameter_to_plot, file_name)
    if showlegend == "y":
        axes[0, 0].legend(loc="upper right", shadow=True, fancybox=True, fontsize=8)
    
    i += 1

# Allow the input screen instead of the graph
plt.ioff()


# i is the object number    
i = 0
sum_mass = 0

object_number = 0
for (file_path, file_name) in file_array:  
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
        file_name_dataframe_path = os.path.abspath(os.path.join(data_output_folder_path + '/' + file_name_dataframe + '/' ))   
        
        graph_dataZoomed = pd.read_csv(file_name_dataframe_path, header=0)

        # Find minimum and maximum for trendlines
        if 'y' in collision:
            graph_data_window = trim_from_collision(graph_dataZoomed, calc_file_name_path, trendline_times, trendline_time_counter)
    
        else: 
            graph_data_window = trim(graph_dataZoomed, calc_file_name_path)

        # PFind trendlines and graph
        graph_data_window = best_fit_fun_graph(fig, axes, graph_data_window, line_style_array[i], line_color_array[i], which_parameter_to_plot, mass[object_number],file_name_dataframe_path, calc_file_name_path)
        
        file_name_dataframe_trendline = file_name  + str(s) + "trend.csv"
        file_name_dataframe_path_trendline = os.path.abspath(os.path.join(data_output_folder_path + '/' + file_name_dataframe_trendline + '/' ))   
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
            trendline_check = input ('Do you want to plot another trendline for your data? (n) or (y)')    
            if trendline_check != 'y':
                break

    object_number += 1      
    i += 1
    plt.tight_layout()
    plt.ioff()
    
    
# exit if spacebar or esc is pressed
input('press return to finish')   

# Save the image of the graphs when you close it
image_name= "graph_" + which_parameter_to_plot + ".png"
graph_path = os.path.abspath(os.path.join(data_output_folder_path, image_name ))   
fig.savefig(graph_path)
plt.show()  
plt.ioff()

         
