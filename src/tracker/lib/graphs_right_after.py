# Takes a *.csv file named 3Dposition.csv with the headers: Time, x,y,z
# The program smooths the data, and finds the velocity and acceleration for each direction.
# 9 Graphs are the result. I will continually look for new data as it is collected, but it does have a bit of a lag.
# The first 3 data points will be removed as the smoothing makes them useless.

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

from tracker.lib.user_input import select_type_of_tracking, select_multiple_files
from tracker.lib.graphing import graph_setup, slopes, FindVelandAcc, FindMom, FindKE, FindTotalEnergy, plot_graphs, trim, best_fit_fun, three_D_graphs
from mpl_toolkits import mplot3d

#### MAIN PROGRAM ####
# Setup graph legend

def nine_graphs():
    #TODO data_folder has the radius and mass of the object so it can be used later for momentum
    type_of_tracking, _, data_output = select_type_of_tracking() 
    file_type = '.csv'

    num_files, file_array = select_multiple_files (data_output, file_type)

    mass = 0
    plt.ion()
    line_style_array = ("solid", "dashed", "dotted", "dashdot")
    line_color_array = ("blue", "red", "green", "cyan", "magenta")


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


    fig, axes, points_to_smooth, header_list, fig_3D, axes_3D = graph_setup(
        which_parameter_to_plot
    )


    # Instruction on how to use the graph
    ## It would be nice if this was showing on the screen with the graphs
    print(
        "When you can see your data, zoom into the section you want to see. Then hit backarrow to see it for a longer time."
    )
    print("Figure out what time interval you will want to see")
    print("Close the graph window to save the file")

    i = 0
    for (file_path, file_name) in file_array:
        # From the t,x,y,z create a data_frame that also has velocity and acceleration in all 3 directions

        print(file_path)
        if which_parameter_to_plot != "a":
            mass = float(input("What is the mass of your object?"))
            if which_parameter_to_plot == "E":
                vert_axis = input("Which axis is up (x,y, or z)?")
                zero_ref_for_PE = float(
                    input(
                        "What value in meters do you want your zero reference frame. Note: you might want to say zero on the first round then modify."
                    )
                )

        # Using weighted differences find the velocity and acceleration graphs
        data_frame, graph_data, smooth_data_to_graph = FindVelandAcc(
            file_path, file_name, header_list, points_to_smooth
        )

        # Use the velocity data to find momentum or energy
        if which_parameter_to_plot == "p":
            data_frame, graph_data, smooth_data_to_graph = FindMom(
                data_frame, header_list, points_to_smooth, mass
            )
        if which_parameter_to_plot == which_parameter_to_plot == "E":
            data_frame, graph_data, smooth_data_to_graph = FindKE(
                data_frame, header_list, points_to_smooth, mass
            )
            data_frame, graph_data, smooth_data_to_graph = FindTotalEnergy(
                data_frame, header_list, points_to_smooth, mass, vert_axis, zero_ref_for_PE
            )

        # Save data to a file Time, x,y,z, Vx, Vy, Vz, Ax, Ay, Az
        file_name_dataframe = file_name + "graph.csv"
        file_name_dataframe_path = os.path.abspath(os.path.join(file_path, file_name_dataframe ))   

        smooth_data_to_graph.to_csv(file_name_dataframe_path)

        plot_graphs(
            smooth_data_to_graph,
            line_style_array[i],
            line_color_array[i],
            fig,
            axes,
            which_parameter_to_plot,
            file_name,
        )
        if showlegend == "y":
            axes[0, 0].legend(loc="upper right", shadow=True, fancybox=True, fontsize=8)
        # The graph will animate every 500 milliseconds or 0.5 seconds.
        # We might be able to reduce this to get it closer to realtime graphs when they run at the same time.

        # Add a 3D graph
        three_D_graphs(axes_3D, smooth_data_to_graph, line_style_array[i], line_color_array[i])
        i += 1
    ###TO DO THIS DOES NOT WORK. I GAVE UP SINCE THERE IS NOT ALWAYS A POINT ASSOCIATED WITH EACH OBJECT AT ANY GIVEN TIME
    # If the momentum is plotted, we also want the total momentum plotted for each axis
    # if which_parameter_to_plot =='p':
    #    for i,var in enumerate(['x','y','z']):
    #        p_total =[]
    #        p_var = str('P'+ str(var))
    #        p_total_var = str('pTot'+ str(var))
    #        for (filepath, filename) in FileArray:
    #            file_name_graphed =  filename + 'graph.csv'
    #            file_name_graphed= os.path.abspath(os.path.join(basepath + "\\" + file_name_graphed + ".csv"))
    #            graph_data = pd.read_csv(filepathZoomed, header=0, names = headerlist)
    #
    #            smooth_data_to_graph = graph_data.set_index('Time')
    #            p_total = p_total + graph_data[p_var]
    #        smooth_data_to_graph[p_var].plot(ax=axes[2,i], title=p_var, fontsize=8, marker='.', markersize=6, color=LineC, linewidth=1)

    # plt.subplots_adjust(top=0.92,bottom=0.08,left=0.10,right=0.95, w_space=0.55, h_space=0.9)
    plt.show()
    # Save the datafile that has Time, x,y,z, Vx,Vy,Vz, Ax,Ay,Az  as well as differences
    # fig.savefig(filename + '.png')

    # lests the user enter data instead of trying to refresh the plot
    plt.ioff()
    # Ask the user if they want to trim their data.
    i = 0


    headerlistZoomed = ["Time", "x", "y", "z"]
    for (file_path, file_name) in file_array:
        trendline_check = input(
            "Do you want trendlines for your first graph? y or n (case sensitive)"
        )
        if trendline_check in ["y", "Y", "yes", "Yes"]:
            file_name_dataframe = file_name + "graph.csv"
            file_name_dataframe_path = os.path.abspath(os.path.join(file_path, file_name_dataframe ))   
            graph_dataZoomed = pd.read_csv(file_name_dataframe_path, header=0)

            Graph_data_window = trim(graph_dataZoomed)

            # PlotGraphs(Graph_data_window, LineStyleArray[i], LineColorArray[i], fig, axes)
            best_fit_fun(Graph_data_window, line_style_array[i], which_parameter_to_plot)
        i += 1
    time.sleep(1)

    # Turns the plot back on and displays the curve fits
    plt.tight_layout()
    plt.show()

    # Save the image of the graphs when you close it
    image_name= file_name + "section.png"
    file_name_dataframe_path = os.path.abspath(os.path.join(data_output, image_name ))   

    fig.savefig(file_name_dataframe_path)
