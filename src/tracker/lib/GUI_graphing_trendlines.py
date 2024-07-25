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
from scipy import linalg
from scipy.signal import lfilter
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import os
import time
from PyQt5.QtWidgets import *

import numpy as np
import csv

from tracker.lib.general import GUI_creates_an_array_of_csv_files
from tracker.lib.setup_files import create_new_folder, create_calc_file
from tracker.lib.user_input import select_type_of_tracking, select_multiple_files
from tracker.lib.graphing import plot_graphs, GUI_trim, trim_from_collision, three_D_graphs, FindVelandAccTrend, FindMom, FindKE, FindTotalEnergy
from mpl_toolkits import mplot3d

def plot_style_color():
    # Defaults for colors and shapes of lines for graph
    show_legend =  True
    line_style_array = ["solid", "dashed", "dotted", "dashdot"]
    line_color_array = ["red", "blue", "green", "cyan", "magenta"]
    marker_shape_array = ["^","s","o","*","X"]
    return line_style_array, line_color_array, marker_shape_array, show_legend

def objects_graphed_in_their_color(default_color, file_name):
    # Overrides the default colors if the names of the objects are a color
    recognized_colors = ('red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'black', 'white')
    
    # Convert the filename to lowercase to make the search case-insensitive
    filename_lower = file_name.lower()
    
    # Search for each color in the filename
    for color in recognized_colors:
        if color in filename_lower:
            return color  # Return the first found color
    
    return default_color  # Return the original default color if no color is found

#TODO input_folder has the radius and mass of the object so it can be used later for momentum
def GUI_graph (which_parameter_to_plot, data_output_folder_path, graph_color_ranges, csv_files_array, points_to_smooth ):

    # Setup Trendline folder. I
    # If user looks at a different variable or time domain, new folder created.
    for (file_path, file_name, mass) in csv_files_array:
        print(file_path, file_name)
    new_sheet_folder = 'trendlines'
    trendline_folder_path = create_new_folder (file_path, new_sheet_folder)
    #create a file to store the trendline equations 
    

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
        print('\nYour folder will be available:\n', file_name_dataframe_path)    
        smooth_data_to_graph = pd.read_csv(file_name_dataframe_path, header=0)

        # Using weighted differences find the velocity and acceleration graphs
        __, graph_data, smooth_data_to_graph = FindVelandAccTrend(
            smooth_data_to_graph, points_to_smooth
        )
        # Use the velocity data to find momentum or energy
        if which_parameter_to_plot in {"p","P"}:
            __, graph_data, smooth_data_to_graph = FindMom(smooth_data_to_graph,
                smooth_data_to_graph, header_list, points_to_smooth, mass
            )
        if which_parameter_to_plot == "E":
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

def best_fit_fun_graph(fig, axes, Graph_data_window, LineS, LineC, which_parameter_to_plot, mass, file_name_dataframe_path, calc_file_name_path, trendline_type):
# create a trendline with least squares method
    
    # Creates an exponential trendline
    def funexp(x, t, y):    
        return x[0] * np.exp(-x[1] * t + x[2] )

    def funsine(x, t, y): 
        return x[0] * np.sin(x[1] * t + x[2] ) + x[3] - y 
    
    def fun_damped_sine(x, t, y):
        return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t + x[3])  - y
    
    def funlinear(x, t, y):
        return x[0] * t + x[1] - y
    
    def funquadratic(x, t, y):
        return x[0] * t**2 + x[1] * t + x[2] - y
    


    # Generates a curve of best fit function based on the type use specifies in the future
    def generate_data(axes, calc_file_name_path, trendline, Var_for_loop, t, A, sigma, omega, beta, noise=0, n_outliers=0, random_state=0 ):
        A = round(A,5)
        sigma = round(sigma,5)
        omega = round(omega,5)
        beta = round(beta,5)
        if trendline == 'exp':
            trendline_equation = str(Var_for_loop) + '= ' + A + ' * np.exp(-'+ sigma +' * t +'+ omega +  ')'
            trendline_value = A * np.exp(-sigma * t + omega )
        elif trendline == 'sine':
            trendline_equation= str (Var_for_loop) + '= ' + A + '  * sin( ' + sigma + ' * t  + ' + omega, ') + ' + beta
            trendline_value = A * np.sin(sigma * t + omega ) + beta
        elif trendline == 'damped_sine':
            trendline_equation =str (Var_for_loop) + '= ' + A + ' * np.exp(-' + sigma + ' * t) * np.sin( ' +  omega + ' * t + ' + beta
            trendline_value = A * np.exp(-sigma * t) * np.sin(omega * t + beta) 
        elif trendline == 'linear' or trendline == 'l' :
            trendline_equation =str(Var_for_loop) + '= '+ str(A) + ' * t +' + str(sigma)
            trendline_value = A * t + sigma
            omega = 0
            beta = 0
        elif trendline == 'quadratic' or trendline == 'q':
            trendline_equation =str(Var_for_loop) + '= ' + str(A) + '*t^2 + ' + str(sigma) +' * t + ' + str(omega)
            trendline_value = A *t**2 + sigma * t + omega
            beta = 0
        else:
            print(' No trendline given')
            trendline_equation = ''
            trendline_value= 0
        print(trendline_equation)
        with open(calc_file_name_path, 'a') as calcs_to_file:
            calcs_to_file.write(f'{trendline_equation},{A},{sigma},{omega},{beta}\n') 
        return trendline_equation, trendline_value

    #This will find call the functions to generate the data for each graph given the y_axis
    def find_trendline_of_each_graph(axes, trendline, VarForLoop, y_lsq_var, LineS):
        # global graph_data

        vert_data = np.array(Graph_data_window[VarForLoop])

        x0 = np.array([1.0, 1.0, 1.0, 1.0])
        guess_mean = np.mean(vert_data)
        guess_phase = 0
        guess_amp = np.max(vert_data)-np.mean(vert_data)
        guess_freq = 0.5
        xsine0 = np.array([guess_amp, guess_freq, guess_phase, guess_mean])
        # print(xsine0)
        #print(Graph_data_window)
        
        
        #basic least squares
        #res_lsq = least_squares(fun, x0, args=(horiz_data,vert_data)) # need entire columns of time and  each variable
        if trendline == 'exp':
            res_lsq = least_squares(funexp, x0, loss='soft_l1', f_scale=0.1, args=(horiz_data, vert_data))        
        elif trendline == 'sine':
            #t= horiz_data
            #optimize_func = lambda xsine0: xsine0[0] * np.sin(xsine0[1] * t + xsine0[2] ) + xsine0[3]-vert_data
            #res_lsq = least_squares(optimize_func, xsine0)
            res_lsq = least_squares(funsine, xsine0, loss='soft_l1', f_scale=0.1, args=(horiz_data, vert_data))        
        elif trendline == 'damped_sine':
            res_lsq = least_squares(fun_damped_sine, x0, loss='soft_l1', f_scale=0.1, args=(horiz_data, vert_data))        
        elif trendline == 'linear' or trendline == 'l' :
            res_lsq = least_squares(funlinear, x0, loss='soft_l1', f_scale=0.1, args=(horiz_data, vert_data))        
        elif trendline == 'quadratic' or trendline == 'q':
            res_lsq = least_squares(funquadratic, x0, loss='soft_l1', f_scale=0.1, args=(horiz_data, vert_data))        

        return res_lsq

    # BestFitFun Main Program
    for i,var in enumerate(['x','y','z']):
        # After viewing the data the person selects the type of trendline they want to use for each graph. 
        trendline = trendline_type[i]   # 0,1,2 represents x,y, and z position curve fit option
        horiz_data = np.array(Graph_data_window['Time'])
        
        # Find trendline of position data
        y_lsq_var = str(str(var)) 
        
        # Find trendline from position data
        res_lsq  = find_trendline_of_each_graph(axes, trendline, var, y_lsq_var, LineS) 

        ''' An attempt to calculate the error for each parameter in the equation
        J = res_lsq.jac
        cov_inv = J.T.dot(J)
        # Remove rows with zeros
        indices = np.all(cov_inv == 0, axis=1)
        nonzeros = cov_inv[np.logical_not(indices)]
        
        # Remove columns with zeros
        idx = np.argwhere(np.all(nonzeros[..., :] == 0, axis=0))
        cov_inv = np.delete(nonzeros, idx, axis=1)
        print(cov_inv)
        cov = linalg.inv(cov_inv)
        variance = np.sqrt(np.diagonal(cov))
        mse = (res_lsq.fun**2).mean()
        variance = variance * mse
        '''
        mse = (res_lsq.fun**2).mean()
        # dFit = errFit( np.linalg.inv(np.dot(result.jac.T, result.jac)), (errFunc(result.x, xdata, ydata)**2).sum()/(len(ydata)-len(pstart) ) ) 
        # https://stackoverflow.com/questions/40187517/getting-covariance-matrix-of-fitted-parameters-from-scipy-optimize-least-squares

        with open(calc_file_name_path, 'a') as calcs_to_file:
            calcs_to_file.write(f'mean square error,{mse}\n')
        
        trendline_equation, Graph_data_window[y_lsq_var] = generate_data(axes, calc_file_name_path, trendline, var, horiz_data, *res_lsq.x)
        
        A, sigma, omega, beta, = res_lsq.x

        # Find velocity data from the trendline of the position data
        y_lsq_v_var = str('V'+ str(var)) 
        # TODO modify so other types of graphs will work besides quadratic and linear
        # assume the trendline for the position data is either linear or quadratic
        # x(t)=At^2+sigmat+omega then v(t)=2At+sigma
        if trendline == 'quadratic' or trendline == 'q':
            trendline = 'l'
            A = 2*A
        else: 
        # linear
            # if x(t)=At+sigma then v(t)=0t+A
            sigma =  A
            A = 0
        # Put velocity data on the spreadsheet
        trendline_equation, Graph_data_window[y_lsq_v_var] = generate_data(axes, calc_file_name_path, trendline, y_lsq_v_var, horiz_data, A, sigma, omega, beta, calc_file_name_path)

        # Find the momentum data and equation from the velocity data and equation
        if which_parameter_to_plot in {"p","P"}:
            p_var = str('P'+ str(var))
            Graph_data_window[p_var] = Graph_data_window[y_lsq_v_var] * mass
            
            mom_A = A*mass
            mom_sigma = sigma *mass
            trendline_equation,Graph_data_window[p_var] = generate_data(axes, calc_file_name_path, trendline, p_var, horiz_data, mom_A, mom_sigma, omega, beta, calc_file_name_path)
        
        # Find acceleration data from the trendline of the velocity data
        if which_parameter_to_plot in {"a","A"}:
            a_var = str('A'+ str(var))
            y_lsq_a_var = str(str(a_var)) 
            # TODO modify so other types of graphs will work besides quadratic and linear
            
            # assume the trendline for the velocity data was linear 
            # linear position: if V(t)=sigma then A(t)=0
            # quadratic position if v(t)=At+sigma A(t)=A
            sigma =  A
            A = 0
            trendline_equation,Graph_data_window[y_lsq_a_var] = generate_data(axes, calc_file_name_path, trendline, y_lsq_a_var, horiz_data, A, sigma, omega, beta, calc_file_name_path)

        
    # TODO Finish KE and PE for trendlines similar to post processing
    if which_parameter_to_plot == 'E':
        y_lsq_KE_var = str('lsqA'+ str('KE')) 
        trendline = trendline_type[0]
        if trendline !='':
            res_lsq = find_trendline_of_each_graph(axes, trendline, 'KE', y_lsq_KE_var, LineS) 
        y_lsq_PE_var = str('lsqA'+ str('PE')) 
        trendline = trendline_type[1]
        if trendline !='':
            res_lsq = find_trendline_of_each_graph(axes, trendline, 'PE', y_lsq_PE_var, LineS)
        y_lsq_Total_var = str('lsqA'+ str('Total')) 
        trendline = trendline_type[2]
        if trendline !='':
            res_lsq = find_trendline_of_each_graph(axes, trendline, 'Total', y_lsq_Total_var, LineS)

    data_frame = pd.DataFrame(Graph_data_window) 
    #print(smooth_data)
    smooth_data_to_graph = data_frame.set_index('Time')
        
    LineC='k'
    for i,var in enumerate(['x','y','z']):
        # print('just before plotting sytle', LineS)
        y_lsq_var = str(str(var) )
        # black line of best fit
        smooth_data_to_graph[y_lsq_var].plot(ax=axes[0,i], label='lsq', linestyle='-', color=LineC, linewidth=1)
        y_lsq_v_var = str('V'+ str(var)) 
        smooth_data_to_graph[y_lsq_v_var].plot(ax=axes[1,i], label='lsq', linestyle='-', color=LineC, linewidth=1)
        if which_parameter_to_plot in {"a","A"}:
            a_var = str('A'+ str(var))
            y_lsq_a_var = str(str(a_var) ) 
            smooth_data_to_graph[y_lsq_a_var].plot(ax=axes[2,i], label='lsq', linestyle='-', color=LineC, linewidth=1)
        if which_parameter_to_plot in {"p","P"}:
            p_var = str('P'+ str(var) )
            smooth_data_to_graph[p_var].plot(ax=axes[2,i], label='lsq', linestyle='-', color=LineC, linewidth=1)
    if which_parameter_to_plot == 'E':
        smooth_data_to_graph[y_lsq_KE_var].plot(ax=axes[2,0], label='lsq', linestyle='-', color=LineC, linewidth=1)
        smooth_data_to_graph[y_lsq_PE_var].plot(ax=axes[2,1], label='lsq', linestyle='-', color=LineC, linewidth=1)

        smooth_data_to_graph[y_lsq_Total_var].plot(ax=axes[2,2], label='lsq', linestyle='-', color=LineC, linewidth=1)
    
    for graph_num_y in range(3):
        axes[2,graph_num_y].set(xlabel='Time (s)')

    return smooth_data_to_graph



def GUI_graph_trendline (title_of_table, graph_widget):

    # Use the trendline_table_widget values to find the trendlines of the current graph
    # The variables are located starting at row 20
    row_info = 20
    data_output =               title_of_table.item(row_info    ,0).text()
    folder_name =               title_of_table.item(row_info + 1,0).text()
    data_output_folder_path =   title_of_table.item(row_info + 2,0).text()
    csv_files_array_str =           title_of_table.item(row_info + 3,0).text()
    which_parameter_to_plot  =  title_of_table.item(row_info + 4,0).text()
    trendline_folder_path =     title_of_table.item(row_info + 5,0).text()
    axes = graph_widget.axes
    #XXX this is what is causing mulitple trendlines to show up.
    # We need to reference the original figure instead
    fig = graph_widget.fig
    line_style_array, line_color_array, marker_shape_array, show_legend = plot_style_color()
    
    calc_file_name =  "calcs.csv"
    calc_file_name_path = os.path.abspath(os.path.join(trendline_folder_path + '/' + calc_file_name + '/' ))   
    create_calc_file(calc_file_name_path)
    i = 0
    num_objects = title_of_table.columnCount()
    for column in range(num_objects):
        trendline_type = []
        
        # Extract the name from the table without getting the color
        name = title_of_table.item(0,column).text()
        start_idx = name.find('(')
        end_idx = name.find(')')
        if start_idx != -1 and end_idx != -1:
            name_without_parentheses = name[:start_idx] + name[end_idx + 1:]
        else:
            name_without_parentheses = name.strip()
        name = name_without_parentheses

        mass = float(title_of_table.item(1,column).text())
        
        try: x_min = float(title_of_table.item(2,column).text())
        except: x_min = float(title_of_table.item(2,0).text())

        try: x_max = float(title_of_table.item(3,column).text())
        except: x_max = float(title_of_table.item(3,0).text())

            # print('csv_files_array',csv_files_array)
        print(column)
        
        # gets the comboBox located in the cellWidget that shows which trendline they chose
        trendline_type.append(title_of_table.cellWidget(4, column).currentText()) # x
        trendline_type.append(title_of_table.cellWidget(5, column).currentText()) # y
        trendline_type.append(title_of_table.cellWidget(6, column).currentText()) # z

        print (trendline_type, trendline_type[1])
         
        with open(calc_file_name_path, 'a') as calcs_to_file:
            calcs_to_file.write(f'{name}\n') 
        print(name)
        file_name_dataframe = name  + "sheet.csv"
        file_name_dataframe_path = os.path.abspath(os.path.join(trendline_folder_path + '/' + file_name_dataframe + '/' ))   
        
        graph_dataZoomed = pd.read_csv(file_name_dataframe_path, header=0)

        # Find minimum and maximum for trendlines
        graph_data_window = GUI_trim(graph_dataZoomed, calc_file_name_path, x_min, x_max)
        with open(calc_file_name_path, 'a') as calcs_to_file:
            calcs_to_file.write(f'mass, {mass}, kg\n') 

        # If graphing a color, use its color on the graph
        line_color_array[i]= objects_graphed_in_their_color(line_color_array[i], name)

        # Find trendlines and graph
        graph_data_window = best_fit_fun_graph(fig, axes, graph_data_window, line_style_array[i], line_color_array[i], which_parameter_to_plot, mass,file_name_dataframe_path, calc_file_name_path, trendline_type)
        
        file_name_dataframe_trendline = name  + str(i) + "trend.csv"
        file_name_dataframe_path_trendline = os.path.abspath(os.path.join(trendline_folder_path + '/' + file_name_dataframe_trendline + '/' ))   
        graph_data_window.to_csv(file_name_dataframe_path_trendline)       
        
        image_name= "graph_trendlines" + which_parameter_to_plot + ".png"
        graph_path = os.path.abspath(os.path.join(trendline_folder_path, image_name ))   
        fig.savefig(graph_path)
        
        i += 1

    return calc_file_name_path
    
def GUI_show_equations_on_table(title_of_table, calc_file_name_path ):
    first_row = 5
    data =[]
    max_rows = 18
    column = 0
    row_calc_file = -1
    row_table = 1

    ###XXX Does not work!!!
    
    with open(calc_file_name_path) as csvfile:
        
        calc_file = csv.reader(csvfile, delimiter=';')
        for row in calc_file:
            row_calc_file += 1
            row_table += 1
            data.append (row)
            
            if (row_calc_file < first_row):
                    continue
            if (row_calc_file-4)%13 == 0:
                row_table -=12
                column += 1

            ##if  'mean square error' in cell:
            ##    cell = data[1]
            title_of_table.setItem(row_table ,column, QTableWidgetItem(str(row)))

            
            

            
            


