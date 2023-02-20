# Takes a *.csv file named 3Dposition.csv with the headers: Time, x,y,z
# The program smooths the data, and finds the velocity and acceleration for each direction.
# 9 Graphs are the result. 
# Also creates *csv files of the velocity, and acceleration etc. data, then use these for plotting.
# Uses a trimmed data matrix to find trendlines for just the domain in question
# The first 3 data points will be removed as the smoothing makes them useless.

# TODO get the sine and exp trendlines working
# TODO get the energy graphs to work with the GUI



from numpy.lib.twodim_base import mask_indices
import pandas
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib import style


import pyrealsense2 as rs
import cv2
import numpy as np
import pandas as pd
from scipy import *
import os
import time
from scipy.signal import lfilter
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from mpl_toolkits import mplot3d

import requests

def GUI_graph_setup(graph_widget, which_parameter_to_plot):
    # TODO remove when we no longer use smoothing to find velocity and acceleration
    #print('To calculate the velocity, do you want to use 3,5, or 7 point smoothing? \n')
    #PointsToSmooth = int(input('larger smoothing is better for either very high data rate or objects following linear paths'))
    PointsToSmooth = 3                           
    
    #Sets up the graphs
    plt.ion
    #fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(5,5), sharex=True)
    #style.use('seaborn-poster')
    #Smoothing value: The smaller the value the more the smoothing?.
    
    # We need to have a file that has this exact header list. At some point, we can read the header list to see how many objects to graph.
    
    graph_widget.axes[0,0].set(ylabel='Position (m)')
    graph_widget.axes[1,0].set(ylabel='Velocity (m/s)')

    if which_parameter_to_plot =='a':
        graph_widget.axes[2,0].set(ylabel='Acceleration m/s')
    if which_parameter_to_plot =='p':
        graph_widget.axes[2,0].set(ylabel='Momentum (kg*m/s)')
    if which_parameter_to_plot =='E':
        graph_widget.axes[2,0].set(ylabel='Energy (J)')

    for graph_num_y in range(3):
        graph_widget.axes[2,graph_num_y].set(xlabel='Time (s)')
        graph_widget.axes[2,graph_num_y].autoscale(enable=True, axis='both', tight=None)
        graph_widget.axes[2,graph_num_y].xaxis.set_major_locator(ticker.MaxNLocator(5))
        for graph_num_x in range(3):
            graph_widget.axes[graph_num_x,graph_num_y].yaxis.set_major_locator(ticker.MaxNLocator(5))



    return graph_widget, PointsToSmooth




def graph_setup(which_parameter_to_plot):
    # TODO remove when we no longer use smoothing to find velocity and acceleration
    #print('To calculate the velocity, do you want to use 3,5, or 7 point smoothing? \n')
    #PointsToSmooth = int(input('larger smoothing is better for either very high data rate or objects following linear paths'))
    PointsToSmooth = 3                           
    
    #Sets up the graphs
    plt.ion
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(5,5), sharex=True)
    style.use('seaborn-poster')
    #Smoothing value: The smaller the value the more the smoothing?.
    
    # We need to have a file that has this exact header list. At some point, we can read the header list to see how many objects to graph.
    headerlist = ['Time', 'x', 'y', 'z']
    
    axes[0,0].set(ylabel='Position (m)')
    axes[1,0].set(ylabel='Velocity (m/s)')

    if which_parameter_to_plot =='a':
        axes[2,0].set(ylabel='Acceleration m/s')
    if which_parameter_to_plot =='p':
        axes[2,0].set(ylabel='Momentum (kg*m/s)')
    if which_parameter_to_plot =='E':
        axes[2,0].set(ylabel='Energy (J)')

    for graph_num_y in range(3):
        axes[2,graph_num_y].set(xlabel='Time (s)')
        axes[2,graph_num_y].autoscale(enable=True, axis='both', tight=None)
        axes[2,graph_num_y].xaxis.set_major_locator(ticker.MaxNLocator(5))
        for graph_num_x in range(3):
            axes[graph_num_x,graph_num_y].yaxis.set_major_locator(ticker.MaxNLocator(5))

    # Setup 3D Graph
    fig_3D = plt.figure()
    axes_3D = plt.axes(projection='3d')

    axes_3D.set(zlabel='y')     # Switched the y and z as most of the time the camera has y vertical as it is set on a table facing outward
    axes_3D.set_xlabel('x (meters)', labelpad=25)
    axes_3D.set_ylabel('z (meters)', labelpad=25) 
    axes_3D.set_zlabel('y (meters)', labelpad=20)

    return fig, axes, PointsToSmooth, headerlist, fig_3D, axes_3D


def slopes(time_data, _data, PointsToSmooth):
    # Finds the slopes around each point similar to the way Vernier graphical analysis does it
    derivative_data = []
    num_of_points = len(time_data)
    #print('length of time data', num_of_points)
    i = 0
    while i+1 <= num_of_points:
        # If i is to small for point smoothing to work 
        if i+1 < (PointsToSmooth+1)/2:
            if i == 0:                  # first data point             
                derivative_data = np.append(derivative_data,0)
            else:
                # find the slope about the point
                slope = (_data[i+1] -_data[i-1] )/(time_data[i+1]-time_data[i-1])
                derivative_data = np.append(derivative_data,slope)

        # If i is to large for point smoothing to work 
        elif i+1 > num_of_points - ((PointsToSmooth-1)/2):
            if i+1 == num_of_points:    # Last data point
                # The last point will be declared the same as the one before
                derivative_data = np.append(derivative_data,0)
            else:
                # find the slope about the point
                slope = (_data[i] -_data[i-1] )/(time_data[i]-time_data[i-1])
                derivative_data = np.append(derivative_data,slope)
        # point smooth for the rest of the data. The points closer to the point are weighted
        else:
            if PointsToSmooth == 3:
                slope = (_data[i+1] -_data[i-1] )/(time_data[i+1]-time_data[i-1])
                derivative_data = np.append(derivative_data,slope)
            if PointsToSmooth == 5:
                slope = ( 2*((_data[i+1] -_data[i-1] )/(time_data[i+1]-time_data[i-1])) + ((_data[i+2] -_data[i-2] )/(time_data[i+2]-time_data[i-2])) ) /3
                derivative_data = np.append(derivative_data,slope)
            if PointsToSmooth == 7:
                slope = ( 3*((_data[i+1] -_data[i-1] )/(time_data[i+1]-time_data[i-1])) + 2*((_data[i+2] -_data[i-2] )/(time_data[i+2]-time_data[i-2])) + ((_data[i+3] -_data[i-3] )/(time_data[i+3]-time_data[i-3])) )/6
                derivative_data = np.append(derivative_data,slope)
        i +=1
    derivative_data[0]=derivative_data[1]
    derivative_data[-1]=derivative_data[-2]
    return derivative_data

def FindVelandAccTrend(graph_data, PointsToSmooth):
    ## realtime graph as the animate function will repeat every so often. 
    
        
    for i,var in enumerate(['x','y','z']):
        # find the velocity using points further away such as 3 point smoothing velocity = x(i+1)-x(i-1)/(t(i+1)-t(i-1)
        # names the column header as well
        v_var = str('V'+ str(var)) 
        time_data = np.array(graph_data['Time'])
        _data = np.array(graph_data[var])

        # Find the velocity data
        derivative_data = slopes(time_data, _data, PointsToSmooth)
        
        graph_data[v_var] = derivative_data    
    
    for i,var in enumerate(['x','y','z']):
        # find the acceleration using points further away such as 3 point smoothing acceleration = v(i+1)-v(i-1)/(t(i+1)-t(i-1)
        # names the column header as well
        v_var = str('V'+ str(var)) 
        a_var = str('A'+ str(var)) 
        
        _data = np.array(graph_data[v_var])
   
        # Find the acceleration data
        graph_data[a_var] = slopes(time_data, _data, PointsToSmooth)

    data_frame = pd.DataFrame(graph_data) 
    smooth_data_to_graph = data_frame.set_index('Time')

    # The difference between smooth_data and smooth_data_toi_graph is that the 'Time' is indexed in the later
    return data_frame,graph_data, smooth_data_to_graph


def FindVelandAcc(filepath, file_name, headerlist, PointsToSmooth):
    ## realtime graph as the animate function will repeat every so often. 
    
    # Creates a global variable so it can write to the file after someone closes the graph.


    # Reads the current version of the *csv file and smooths it
    path_to_file = os.path.abspath(os.path.join(filepath, file_name))
    graph_data = pd.read_csv(path_to_file, header=0, names = headerlist)
        
    for i,var in enumerate(['x','y','z']):
        # find the velocity using points further away such as 3 point smoothing velocity = x(i+1)-x(i-1)/(t(i+1)-t(i-1)
        # names the column header as well
        v_var = str('V'+ str(var)) 
        time_data = np.array(graph_data['Time'])
        _data = np.array(graph_data[var])

        # Find the velocity data
        derivative_data = slopes(time_data, _data, PointsToSmooth)
        
        graph_data[v_var] = derivative_data    
    
    for i,var in enumerate(['x','y','z']):
        # find the acceleration using points further away such as 3 point smoothing acceleration = v(i+1)-v(i-1)/(t(i+1)-t(i-1)
        # names the column header as well
        v_var = str('V'+ str(var)) 
        a_var = str('A'+ str(var)) 
        
        _data = np.array(graph_data[v_var])
   
        # Find the acceleration data
        graph_data[a_var] = slopes(time_data, _data, PointsToSmooth)

    data_frame = pd.DataFrame(graph_data) 
    smooth_data_to_graph = data_frame.set_index('Time')

    # The difference between smooth_data and smooth_data_toi_graph is that the 'Time' is indexed in the later
    return data_frame,graph_data, smooth_data_to_graph

def FindMom(graph_data, data_frame, headerlist, PointsToSmooth, mass):
    ## realtime grasp as the animate function will repeat every so often. 
    
    # Creates a global variable so it can write to the file after someone closes the graph. TOTO DOES THIS STILL WORK NOW THAT graph_data isn't global?
      
  
    for i,var in enumerate(['x','y','z']):
        # find the momentum  PE KE and TE given the velocity
        # names the column header as well
        v_var = str('V'+ str(var)) 
   
        # find momentum data
        p_var = str('P' + str(var))
        graph_data[p_var] = graph_data[v_var] * mass

    # find Kinetic Energy data
    #graph_data['KE']=0.5 * mass * pd.sum (graph_data['Vx']**2, graph_data['Vy']**2, graph_data['Vz']**2) 
    # graph_data['KE']=0.5 * mass * pd.sum (power(graph_data['Vx'],2), power(graph_data['Vy'],2), power(graph_data['Vz'],2) )

    data_frame = pd.DataFrame(graph_data) 
#    smooth_data_to_graph = data_frame.set_index('Time')
    smooth_data_to_graph = data_frame
    # The difference between smooth_data and smooth_data_toi_graph is that the 'Time' is indexed in the later
    return data_frame, graph_data, smooth_data_to_graph

def FindKE(graph_data, data_frame, headerlist, PointsToSmooth, mass):
    ## realtime graph as the animate function will repeat every so often. 
   
    # find Kinetic Energy data KE =0.5mv^2      where v=  sqrt((Vx^2+Vy^2+Vz^2))
     # .pow(2) says to take the value and raise it to the 2nd power in our case, thus square the entire column of values.
    graph_data['Vx^2'] = graph_data['Vx']**2
    graph_data['Vy^2'] = graph_data['Vy']**2
    graph_data['Vz^2'] = graph_data['Vz']**2
    column_list = ['Vx^2', 'Vy^2', 'Vz^2']
    graph_data['KE'] = 0.5 * mass * graph_data[column_list].sum(axis=1)

    #graph_data['KE'] = 0.5 * mass * graph_data.sum( graph_data['Vx'].pow(2),  graph_data['Vy'].pow(2), graph_data['Vz'].pow(2) )

    # find Kinetic Energy data
    #graph_data['KE']=0.5 * mass * pd.sum (graph_data['Vx']**2, graph_data['Vy']**2, graph_data['Vz']**2) 
    data_frame = pd.DataFrame(graph_data) 
    smooth_data_to_graph = data_frame.set_index('Time')

    # The difference between smooth_data and smooth_data_toi_graph is that the 'Time' is indexed in the later
    return data_frame,graph_data, smooth_data_to_graph

def FindTotalEnergy(graph_data, headerlist, PointsToSmooth, mass, vert_axis, zero_ref_for_PE):
    ## realtime graph as the animate function will repeat every so often. 
   
    # find Total Energy by finding PE and adding it to KE
    
    graph_data['PE'] = mass * 9.81 * (graph_data[vert_axis]-zero_ref_for_PE)

    column_list = ['KE', 'PE']
    graph_data['Total'] = graph_data[column_list].sum(axis=1)

    #graph_data['KE'] = 0.5 * mass * graph_data.sum( graph_data['Vx'].pow(2),  graph_data['Vy'].pow(2), graph_data['Vz'].pow(2) )

    # find Kinetic Energy data
    #graph_data['KE']=0.5 * mass * pd.sum (graph_data['Vx']**2, graph_data['Vy']**2, graph_data['Vz']**2) 
    data_frame = pd.DataFrame(graph_data) 
    smooth_data_to_graph = data_frame.set_index('Time')

    # The difference between smooth_data and smooth_data_toi_graph is that the 'Time' is indexed in the later
    return data_frame,graph_data, smooth_data_to_graph



def plot_graphs(smooth_data_to_graph, LineS,LineC,marker_shape, fig, axes, which_parameter_to_plot, filename):
   # plot all graphs x,y,z,Vx,Vy,Vz,Ax,Ay,Az  
    
    for i,var in enumerate(['x','y','z']):
        #plot graphs
        smooth_data_to_graph[var].plot(ax=axes[0,i], title=var, fontsize=8, marker=marker_shape, markersize=3, color=LineC,  linestyle=LineS, linewidth=0, label=filename)
        v_var = str('V'+ str(var)) 
        smooth_data_to_graph[v_var].plot(ax=axes[1,i], title=v_var, fontsize=8, marker=marker_shape, markersize=3, color=LineC, linestyle=LineS, linewidth=0)
        if which_parameter_to_plot == 'a':
            a_var = str('A'+ str(var)) 
            smooth_data_to_graph[a_var].plot(ax=axes[2,i], title=a_var, fontsize=8, marker=marker_shape, markersize=3, color=LineC, linestyle=LineS, linewidth=0)
        elif which_parameter_to_plot =='p':
            ##if 'CM' not in filename:
            p_var = str('P'+ str(var)) 
            smooth_data_to_graph[p_var].plot(ax=axes[2,i], title=p_var, fontsize=8, marker=marker_shape, markersize=3, color=LineC, linestyle=LineS, linewidth=0)
    if which_parameter_to_plot =='E':
        smooth_data_to_graph['KE'].plot(ax=axes[2,0], title='KE', fontsize=8, marker=marker_shape, markersize=3, color=LineC, linestyle=LineS, linewidth=0)        
        smooth_data_to_graph['PE'].plot(ax=axes[2,1], title='PE', fontsize=8, marker=marker_shape, markersize=3, color=LineC, linestyle=LineS, linewidth=0)
        smooth_data_to_graph['Total'].plot(ax=axes[2,2], title='Total E', fontsize=8, marker=marker_shape, markersize=3, color=LineC, linestyle=LineS, linewidth=0)
    

    
    for count_i in range(3):
        for count_j in range(3):
            axes[count_i,count_j].minorticks_on()   
            if count_i==2: 
                axes[count_i,count_j].set(xlabel='Time (s)') 

def plot_position_graphs(smooth_data_to_graph, LineS,LineC, marker_shape, fig, axes, which_parameter_to_plot, filename):
   # plot all graphs x,y,z,Vx,Vy,Vz,Ax,Ay,Az  

    for i,var in enumerate(['x','y','z']):
        #plot graphs
        smooth_data_to_graph[var].plot(ax=axes[0,i], title=var, fontsize=8, marker=marker_shape, markersize=3, color=LineC,  linestyle=LineS, linewidth=1, label=filename)
        
    for graph_num_y in range(3):
        axes[2,graph_num_y].set(xlabel='Time (s)')

def GUI_trim(graph_data, calc_file_name_path, xmin, xmax ):
    #  Trims the data to find the trendline
    with open(calc_file_name_path, 'a') as calcs_to_file:
        calcs_to_file.write(f'x_min, {xmin}, seconds \n') 
        calcs_to_file.write(f'x_max, {xmax}, seconds \n')    

    Graph_data_window = graph_data[graph_data["Time"]>xmin]
    Graph_data_window = Graph_data_window[Graph_data_window["Time"]<xmax]
    
    return Graph_data_window

def trim(graph_data, calc_file_name_path ):
    #  Trims the data to find the trendline
    
    xmin = float(input ('What is the minimum time(seconds)  xmin='))
    xmax = float(input ('What is the maximum time(seconds)  xmax='))
    with open(calc_file_name_path, 'a') as calcs_to_file:
        calcs_to_file.write(f'x_min, {xmin}, seconds \n') 
        calcs_to_file.write(f'x_max, {xmax}, seconds \n')
    

    Graph_data_window = graph_data[graph_data["Time"]>xmin]
    Graph_data_window = Graph_data_window[Graph_data_window["Time"]<xmax]
    
    return Graph_data_window

def trim_from_collision(graph_data, calc_file_name_path, trendline_times, trendline_time_counter):
    # trims the data to find the trendline from a collision 

    xmin = trendline_times[trendline_time_counter]
    xmax = trendline_times[trendline_time_counter+1]
    print('xmin, xmax', xmin, xmax)
    with open(calc_file_name_path, 'a') as calcs_to_file:
        calcs_to_file.write(f'x_min, {xmin}, seconds \n') 
        calcs_to_file.write(f'x_max, {xmax}, seconds \n')
    

    Graph_data_window = graph_data[graph_data["Time"]>xmin]
    Graph_data_window = Graph_data_window[Graph_data_window["Time"]<xmax]
    
    return Graph_data_window
    
def best_fit_fun(axes, Graph_data_window, LineS, which_parameter_to_plot, calc_file_name_path):
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
            trendline_equation = str(Var_for_loop+ '= ' + A + ' * np.exp(-'+ sigma +' * t +'+ omega +  ')')
            trendline_value = A * np.exp(-sigma * t + omega )
        elif trendline == 'sine':
            trendline_equation= str (Var_for_loop + '= ' + A + '  * sin( ' + sigma + ' * t  + ' + omega, ') + ' + beta)
            trendline_value = A * np.sin(sigma * t + omega ) + beta
        elif trendline == 'damped_sine':
            trendline_equation =str (Var_for_loop + '= ' + A + ' * np.exp(-' + sigma + ' * t) * np.sin( ' +  omega + ' * t + ' + beta)
            trendline_value = A * np.exp(-sigma * t) * np.sin(omega * t + beta) 
        elif trendline == 'linear' or trendline == 'l' :
            trendline_equation =str(Var_for_loop) + '= '+ str(A) + ' * t +' + str(sigma)
            trendline_value = A * t + sigma
        elif trendline == 'quadratic' or trendline == 'q':
            trendline_equation = str(A) + '*t^2 + ' + str(sigma) +' * t + ' + str(omega)
            trendline_value = A *t**2 + sigma * t + omega
        else:
            print(' No trendline given')
            trendline_equation = ''
            trendline_value= 0
        print(trendline_equation)
        with open(calc_file_name_path, 'a') as calcs_to_file:
            calcs_to_file.write(f'{trendline_equation},{A},{sigma},{omega},{beta}\n') 
        return trendline_equation, trendline_value



    #This will find call the functions to generate the data for each graph given the y_axis
    def find_trendline_of_each_graph(axes, trendline, VarForLoop, y_axis_lsq, LineS):
        # global graph_data

        vert_data = np.array(Graph_data_window[VarForLoop])

        x0 = np.array([1.0, 1.0, 1.0, 1.0])
        guess_mean = np.mean(vert_data)
        guess_phase = 0
        guess_amp = np.max(vert_data)-np.mean(vert_data)
        guess_freq = 0.5
        xsine0 = np.array([guess_amp, guess_freq, guess_phase, guess_mean])
        print(xsine0)
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
        
        
        
        #robust handels outliers better?
        #print (*res_lsq.x)

        # Generates a set of data for the curve of best fit
        trendline_equation, Graph_data_window[y_axis_lsq] = generate_data(axes, calc_file_name_path, trendline, VarForLoop, horiz_data, *res_lsq.x)
    # BestFitFun Main Program
    for i,var in enumerate(['x','y','z']):
        # After viewing the data the person selects the type of trendline they want to use for each graph. 
        print('What trendline would you like? linear (l) or quadratic(q)? Hit enter to skip trendline')
        #If 'enter' then goes back to the main program
        
       
        horiz_data = np.array(Graph_data_window['Time'])
        
        y_lsq_var = str('lsq'+ str(var)) 
        print ('For the ',var, ' vs time graph?') 
        trendline = input()
        if trendline != '':
            find_trendline_of_each_graph(axes, trendline, var, y_lsq_var, LineS) 
        
        v_var = str('V'+ str(var)) 
        y_lsq_v_var = str('lsqV'+ str(var)) 
        print ('For the ',v_var, ' vs time graph?') 
        trendline = input()
        if trendline != '':
            find_trendline_of_each_graph(axes, trendline,v_var, y_lsq_v_var, LineS) 

        if which_parameter_to_plot == 'a':
            a_var = str('A'+ str(var))
            y_lsq_a_var = str('lsqA'+ str(var)) 
            print ('For the ',a_var, ' vs time graph?') 
            trendline = input()
            if trendline !='':
                find_trendline_of_each_graph(axes, trendline, a_var, y_lsq_a_var, LineS) 
    
        if which_parameter_to_plot == 'p':
                p_var = str('P'+ str(var))
                y_lsq_p_var = str('lsqA'+ str(var)) 
                print ('For the ',p_var, ' vs time graph?') 
                trendline = input()
                if trendline !='':
                    find_trendline_of_each_graph(axes, trendline, p_var, y_lsq_p_var, LineS) 

    if which_parameter_to_plot == 'E':
        y_lsq_KE_var = str('lsqA'+ str('KE')) 
        print ('For the KE vs time graph?') 
        trendline = input()
        if trendline !='':
            find_trendline_of_each_graph(axes, trendline, 'KE', y_lsq_KE_var, LineS) 
        y_lsq_PE_var = str('lsqA'+ str('PE')) 
        print ('For the PE vs time graph?') 
        trendline = input()
        if trendline !='':
            find_trendline_of_each_graph(axes, trendline, 'PE', y_lsq_PE_var, LineS)
        y_lsq_Total_var = str('lsqA'+ str('Total')) 
        print ('For the Total Energy vs time graph?') 
        trendline = input()
        if trendline !='':
            find_trendline_of_each_graph(axes, trendline, 'Total', y_lsq_Total_var, LineS)

    data_frame = pd.DataFrame(Graph_data_window) 
    #print(smooth_data)
    smooth_data_to_graph = data_frame.set_index('Time')
    

    for i,var in enumerate(['x','y','z']):
        # print('just before plotting sytle', LineS)
        y_lsq_var = str('lsq'+ str(var)) 
        smooth_data_to_graph[y_lsq_var].plot(ax=axes[0,i], label='lsq', linestyle=LineS, color='y', linewidth=1)
        y_lsq_v_var = str('lsqV'+ str(var)) 
        smooth_data_to_graph[y_lsq_v_var].plot(ax=axes[1,i], label='lsq', linestyle=LineS, color='y', linewidth=1)
        if which_parameter_to_plot == 'a':
            y_lsq_a_var = str('lsqA'+ str(var)) 
            smooth_data_to_graph[y_lsq_a_var].plot(ax=axes[2,i], label='lsq', linestyle=LineS, color='y', linewidth=1)
        if which_parameter_to_plot == 'p':
            y_lsq_p_var = str('lsqA'+ str(var)) 
            smooth_data_to_graph[y_lsq_p_var].plot(ax=axes[2,i], label='lsq', linestyle=LineS, color='y', linewidth=1)
    if which_parameter_to_plot == 'E':
        smooth_data_to_graph[y_lsq_KE_var].plot(ax=axes[2,0], label='lsq', linestyle=LineS, color='y', linewidth=1)
        smooth_data_to_graph[y_lsq_PE_var].plot(ax=axes[2,1], label='lsq', linestyle=LineS, color='y', linewidth=1)
        smooth_data_to_graph[y_lsq_Total_var].plot(ax=axes[2,2], label='lsq', linestyle=LineS, color='y', linewidth=1)
    for graph_num_y in range(3):
        axes[2,graph_num_y].set(xlabel='Time (s)')

def best_fit_fun_graph(fig, axes, Graph_data_window, LineS, LineC, which_parameter_to_plot, mass, file_name_dataframe_path, calc_file_name_path):
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
        elif trendline == 'quadratic' or trendline == 'q':
            trendline_equation =str(Var_for_loop) + '= ' + str(A) + '*t^2 + ' + str(sigma) +' * t + ' + str(omega)
            trendline_value = A *t**2 + sigma * t + omega
        else:
            print(' No trendline given')
            trendline_equation = ''
            trendline_value= 0
        print(trendline_equation)
        with open(calc_file_name_path, 'a') as calcs_to_file:
            calcs_to_file.write(f'{trendline_equation},{A},{sigma},{omega},{beta}\n') 
        return trendline_equation, trendline_value

    #This will find call the functions to generate the data for each graph given the y_axis
    def find_trendline_of_each_graph(axes, trendline, VarForLoop, y_axis_lsq, LineS):
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
        
        
        
        #robust handels outliers better?
        #print (*res_lsq.x)

        # Generates a set of data for the curve of best fit
        trendline_equation, Graph_data_window[y_axis_lsq] = generate_data(axes, calc_file_name_path, trendline, VarForLoop, horiz_data, *res_lsq.x)
        
        return res_lsq.x

    # BestFitFun Main Program
    for i,var in enumerate(['x','y','z']):
        # After viewing the data the person selects the type of trendline they want to use for each graph. 
        print('What trendline would you like? linear (l) or quadratic(q)? Hit enter to skip trendline')
        #If 'enter' then goes back to the main program
        
       
        horiz_data = np.array(Graph_data_window['Time'])
        
        # Find trendline of position data
        y_lsq_var = str(str(var) + '(t)') 
        print ('For the ',var, ' vs time graph?') 
        trendline = input()
        if trendline != '':
            A, sigma, omega, beta = find_trendline_of_each_graph(axes, trendline, var, y_lsq_var, LineS) 

        # Find velocity data from the trendline of the position data
        y_lsq_v_var = str('V'+ str(var) + '(t)') 
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
        trendline_equation,Graph_data_window[y_lsq_v_var] = generate_data(axes, calc_file_name_path, trendline, y_lsq_v_var, horiz_data, A, sigma, omega, beta, calc_file_name_path)

        # Find the momentum data and equation from the velocity data and equation
        if which_parameter_to_plot == 'p':
            p_var = str('P'+ str(var) + '(t)')
            Graph_data_window[p_var] = Graph_data_window[y_lsq_v_var] * mass
            
            mom_A = A*mass
            mom_sigma = sigma *mass
            trendline_equation,Graph_data_window[p_var] = generate_data(axes, calc_file_name_path, trendline, p_var, horiz_data, mom_A, mom_sigma, omega, beta, calc_file_name_path)
        
        # Find acceleration data from the trendline of the velocity data
        if which_parameter_to_plot == 'a':
            a_var = str('A'+ str(var))
            y_lsq_a_var = str(str(a_var) + '(t)') 
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
        print ('For the KE vs time graph?') 
        trendline = input()
        if trendline !='':
            find_trendline_of_each_graph(axes, trendline, 'KE', y_lsq_KE_var, LineS) 
        y_lsq_PE_var = str('lsqA'+ str('PE')) 
        print ('For the PE vs time graph?') 
        trendline = input()
        if trendline !='':
            find_trendline_of_each_graph(axes, trendline, 'PE', y_lsq_PE_var, LineS)
        y_lsq_Total_var = str('lsqA'+ str('Total')) 
        print ('For the Total Energy vs time graph?') 
        trendline = input()
        if trendline !='':
            find_trendline_of_each_graph(axes, trendline, 'Total', y_lsq_Total_var, LineS)

    data_frame = pd.DataFrame(Graph_data_window) 
    #print(smooth_data)
    smooth_data_to_graph = data_frame.set_index('Time')
    #smooth_data_to_graph.to_csv(file_name_dataframe_path)
    
       

    for i,var in enumerate(['x','y','z']):
        # print('just before plotting sytle', LineS)
        y_lsq_var = str(str(var) + '(t)')
        # black line of best fit
        smooth_data_to_graph[y_lsq_var].plot(ax=axes[0,i], label='lsq', linestyle='-', color='k', linewidth=1)
        y_lsq_v_var = str('V'+ str(var) + '(t)') 
        smooth_data_to_graph[y_lsq_v_var].plot(ax=axes[1,i], label='lsq', linestyle='-', color=LineC, linewidth=1)
        if which_parameter_to_plot == 'a':
            a_var = str('A'+ str(var))
            y_lsq_a_var = str(str(a_var) + '(t)') 
            smooth_data_to_graph[y_lsq_a_var].plot(ax=axes[2,i], label='lsq', linestyle='-', color=LineC, linewidth=1)
        if which_parameter_to_plot == 'p':
            p_var = str('P'+ str(var) + '(t)')
            smooth_data_to_graph[p_var].plot(ax=axes[2,i], label='lsq', linestyle='-', color=LineC, linewidth=1)
    if which_parameter_to_plot == 'E':
        smooth_data_to_graph[y_lsq_KE_var].plot(ax=axes[2,0], label='lsq', linestyle='-', color=LineC, linewidth=1)
        smooth_data_to_graph[y_lsq_PE_var].plot(ax=axes[2,1], label='lsq', linestyle='-', color=LineC, linewidth=1)

        smooth_data_to_graph[y_lsq_Total_var].plot(ax=axes[2,2], label='lsq', linestyle='-', color=LineC, linewidth=1)
    
    for graph_num_y in range(3):
        axes[2,graph_num_y].set(xlabel='Time (s)')

    return smooth_data_to_graph

# Graph the position in 3D
def three_D_graphs(axes_3D, smooth_data_to_graph,LineStyle, linecolor, marker_shape):
    axes_3D.plot3D(smooth_data_to_graph['x'],smooth_data_to_graph['z'], smooth_data_to_graph['y'], linecolor, markersize = 3, marker = marker_shape)
    axes_3D.scatter3D(smooth_data_to_graph['x'],smooth_data_to_graph['z'], smooth_data_to_graph['y'], c=linecolor, cmap=linecolor, marker = marker_shape)

