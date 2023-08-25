## Main program for object tracking ##
## Used pyimagesearch.com for base code ##
## Put the *exe in a folder. with Data folder containing color_i and color_o folders
## Used QtDesigner app

from typing import List, Optional, Tuple
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import *

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg          # In the future we should have the pyqtgraph locally stored. see discord redources
from pyqtgraph.ptime import time

import webbrowser
import ast
import os
import sys
import time
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT 
from mpl_toolkits import mplot3d

from tracker.lib.general import find_objects_to_graph
from tracker.lib.user_input import make_new_folder
from tracker.lib.GUI_tracker import GUI_tracking, find_lower_upper_bounds_on_screen
from tracker.lib.GUI_real_time_color_tracker import GUI_real_time_color_tracking
from tracker.lib.GUI_graphing_trendlines import GUI_graph, GUI_graph_trendline, plot_style_color, GUI_show_equations_on_table
from tracker.lib.graphing import GUI_graph_setup, three_D_graphs, plot_graphs, GUI_trim
from tracker.lib.intel_realsense_D435i import record_bag_file, find_and_config_device, read_bag_file_and_config
from tracker.lib.GUI_library import reload_table
from tracker.lib.color import choose_or_create_color_range

## TODO possibly use this instead of passing each individual piece
class folder_info:
  def __init__(self, types_of_streams_saved, type_of_tracking, input_folder, output_folder, data_output_folder, data_output_folder_path, color_ranges ):
    self.types_of_streams_saved = types_of_streams_saved    # cd: color, id: infrared or id300
    self.type_of_tracking = type_of_tracking                # color or infrared
    self.input_folder = input_folder                        # example: color_i
    self.output_folder = output_folder                      # example: color_o
    self.data_output_folder = data_output_folder            # example: drop1
    self.data_output_folder_path = data_output_folder_path  

    # numpy with objects you are tracking with object name, mass, and radius
    self.color_ranges = color_ranges                      #(lower,upper, color_name, radius_meters, mass)

class image_option:
    def __init__(self, show_RGB, save_RGB, show_depth, save_depth, show_mask, save_mask, save_video):
        self.show_RGB = show_RGB 
        self.save_RGB = save_RGB
        self.show_depth = show_depth
        self.save_depth = save_depth
        self.show_mask = show_mask
        self.save_mask = save_mask
        self.save_video = save_video

class parameter_to_plot:
    def __init__(self, acceleration, momentum, energy):
        if acceleration:
            self.which_parameter_to_plot =  'A'
        elif momentum:
            self.which_parameter_to_plot =  'P'
        elif energy:
            self.which_parameter_to_plot =  'E'

class mlpcanvas(FigureCanvasQTAgg):

    fig: Figure
    axes: np.ndarray

    def __init__(self):
        self.fig, self.axes = plt.subplots(nrows=3, ncols=3, figsize=(10,4.5), sharex=True)
        plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.7, hspace=0.7)
        super(mlpcanvas, self).__init__(self.fig)

    def clear(self):
        del self.axes
        self.fig.clear()
        self.axes = self.fig.subplots(nrows=3, ncols=3, sharex=True, sharey=False,
                       squeeze=True, subplot_kw=None,
                       gridspec_kw=None)

'''class mlpcanvas_3D(FigureCanvasQTAgg):
    def __init__(self):
        # Setup 3D Graph
        fig_3D = plt.figure()
        self.axes_3D = plt.axes(projection='3d')

        self.axes_3D.set(zlabel='y')     # Switched the y and z as most of the time the camera has y vertical as it is set on a table facing outward
        self.axes_3D.set_xlabel('x (meters)', labelpad=25)
        self.axes_3D.set_ylabel('z (meters)', labelpad=25) 
        self.axes_3D.set_zlabel('y (meters)', labelpad=20)

        super(mlpcanvas_3D, self).__init__(fig_3D)
'''       

class MyGUI(QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        ## TODO before making *.exe file !!
        ## TODO move data file to src???
        
        # self.sc = mlpcanvas()
        # self.sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        # self.setCentralWidget(self.sc)
        
        # self.show()
        GUI_file_path = os.path.abspath(os.path.join('src', 'tracker', 'lib', 'GUI_Base.ui'))
        uic.loadUi(GUI_file_path, self)

        self.toolbar=False

        self.DataGraph.setHidden(True)
        
        # Folder Options Hidden
        self.folder_list.setHidden(True)        # List of folders to put the data into
        self.folder_name.setHidden(True)  
        self.table_widget_color.setHidden(True)
        self.folder_name_objects.setHidden(True)
        self.table_widget_color_2.setHidden(True) # For new colors
        self.lineEdit_define_color_name.setHidden(True)
        self.find_lower_upper_button.setHidden(True)
        # Acceleration, Energy, Momentum  Hidden
        self.select_acceleration.setHidden(False)
        self.select_momentum.setHidden(False)
        ## TODO have energy as well
        self.select_energy.setHidden(True)
        # Momentum/Energy Options Hidden
        self.mass_of_object.setHidden(True)
        self.up_axis.setHidden(True)            # If using energy need the axis representing height for E=mgh
        self.r_height.setHidden(True)           # reference height for mgh
        self.save_height_mass.setHidden(True)
        # Color choices hidden (text)
        self.select_default_colors.setHidden(True)
        self.select_your_own_colors.setHidden(True)
        self.define_colors.setHidden(True)
        self.combo_box_objects.setHidden(True)
        # GUI shows up
        self.show()
        # Buttons: Main
        self.find_lower_upper_button.clicked.connect(self.find_lower_upper_bounds)
        self.real_time_button.clicked.connect(self.run_real_time)
        self.tracker_button.clicked.connect(self.run_tracker)
        self.graph_button.clicked.connect(self.run_graph)
        self.Button3DGraph.setHidden(True)
        self.Button3DGraph.clicked.connect(self.run_3D_graph)
        self.trendline_table_widget.setHidden(True)
        self.find_trendlines_button.setHidden(True)
        self.find_trendlines_button.clicked.connect(self.find_trendlines_button_pressed)
        
        
        self.record_bag_button.clicked.connect(self.record_bag)
        self.track_from_bag_button.clicked.connect(self.track_from_bag)
        # Buttons: Minor
        self.color_button.clicked.connect(self.color_button_pressed)
        self.infrared_90_button.clicked.connect(self.infrared_90_button_pressed)
        self.other_folder.clicked.connect(self.set_folder_to_other)
        self.skeletal_button.clicked.connect(self.skeletal_tracker_pressed)

        self.save_height_mass.clicked.connect(self.reference_height_save)
        # Radio Buttons: Color Choice 
        self.select_default_colors.toggled.connect(self.default_colors_shown)
        self.select_your_own_colors.toggled.connect(self.your_own_colors_shown)
        self.define_colors.toggled.connect(self.define_colors_shown)
        # Radio Button: 3rd variable graphed
        self.select_acceleration.toggled.connect(self.acceleration_chosen)
        self.select_momentum.toggled.connect(self.momentum_chosen)
        self.select_energy.toggled.connect(self.energy_chosen)
        # Graph info
        self.graph_widget = mlpcanvas()
        self.grid_layout.addWidget(self.graph_widget,0,0,alignment=Qt.Alignment())        
        # self.graph_widget_3D = mlpcanvas_3D()
        self.addToolBar(NavigationToolbar2QT( self.graph_widget , self ))


        # Creating Radio Button Groups so only one of each can be selected
        self.color_group = QButtonGroup()
        self.ame_group = QButtonGroup()
        # Putting buttons in color group
        self.color_group.addButton(self.select_default_colors)
        self.color_group.addButton(self.select_your_own_colors)
        self.color_group.addButton(self.define_colors)
        # Putting buttons in AME group
        self.ame_group.addButton(self.select_acceleration)
        self.ame_group.addButton(self.select_momentum)
        self.ame_group.addButton(self.select_energy)
        # Putting buttons in camera (cam) group

        # Text/Line Edits

        self.folder_name.returnPressed.connect(self.user_creating_folder)
        # Threaded Class Stuff
        '''
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.camera_options)
        self.Worker1.start()
        

    
    This was to select the camera for chosing a color
    def camera_options(self):
        ## TODO If we have to save these, put them somewhere that is gitignored and uses + to define path
        ## TODO show image, I assume as a *.jpg
        cwd = os.getcwd()
        self.cam_0.setPixmap(QtGui.QPixmap(f"{cwd}/cam_img_0.png"))
        self.cam_1.setPixmap(QtGui.QPixmap(f"{cwd}/cam_img_1.png"))
        self.cam_2.setPixmap(QtGui.QPixmap(f"{cwd}/cam_img_2.png"))
        self.C0_Button.setHidden(False)
        self.C1_Button.setHidden(False)
        self.C2_Button.setHidden(False)
        if os.path.isfile(f"{cwd}/cam_img_0.png"):
            os.remove(f"{cwd}/cam_img_0.png")
        if os.path.isfile(f"{cwd}/cam_img_1.png"):
            os.remove(f"{cwd}/cam_img_1.png")
        if os.path.isfile(f"{cwd}/cam_img_2.png"):
            os.remove(f"{cwd}/cam_img_2.png")
'''
    # List Folders to save the file to
    def list_folders(self, data):
        self.folder_list.clear()
        base_path = os.getcwd()
        dir_path = os.path.abspath(os.path.join(base_path, 'data', data, ''))
        # print('data file path', dir_path)
        for f in os.listdir(dir_path):
            self.folder_list.addItem(f)

    # User Created Folder
    def select_object_file(self,data):
        self.combo_box_objects.setHidden(False)
        self.table_widget_color.setHidden(True)
        base_path = os.getcwd()
        dir_path = os.path.abspath(os.path.join(base_path, 'data', data, ''))
        # print('data file path', dir_path)
        list_files = []
        list_files.append('-Select your own color pallet-')
        list_files.append('Create new set of objects')
        for f in os.listdir(dir_path):
            list_files.append(f)
        self.combo_box_objects.addItems(list_files)

    def user_creating_folder(self):
        data = "color_o"
        base_path = os.getcwd()
        dir_path = os.path.abspath(os.path.join(base_path, 'data', data, ''))
        check = False
        x = 0

        while not check:
            if 'o' in data:
                data_output_folder = self.folder_name.text()
            elif 'i' in data:
                data_output_folder = self.folder_name_objects.text()
            data_output_folder_path = os.path.abspath(os.path.join(dir_path, data_output_folder, ''))
            # Create a folder to store all of the data in if it does not already exist
            if not os.path.exists(data_output_folder_path):
                os.makedirs(data_output_folder_path)
                check = True
            else:
                x = 1
                print("If Tracking, this will OVERWRITE your existing folder. If graphing, just hit graph")
                break
        if x == 0:
            print(f'\nYour folder will be available:\n {data_output_folder_path}')
        else:
            pass
        # return data_output_folder, data_output_folder_path

    # Infrared chosen (vs. color)
    def infrared_90_button_pressed(self):
        # Define the the folders that will be used
        type_of_tracking = 'infrared'
        input_folder = 'infrared_i'
        data_output = 'infrared_o'
        # Color options are hidden
        self.select_default_colors.setHidden(True)
        self.select_your_own_colors.setHidden(True)
        # show other options
        self.folder_list.setHidden(False)
        self.folder_name.setHidden(False)
        self.select_acceleration.setHidden(False)
        self.select_momentum.setHidden(False)
        '''
        self.select_energy.setHidden(False)
        '''
        self.select_energy.setHidden(True)

        # Folder List Showing Folders
        self.list_folders(data_output)
        print('Future')
        
    def set_folder_to_other(self):
        type_of_tracking = 'other'
        input_folder = 'other_i'
        data_output = 'other_o' 
        # Select the color from a list, use a predefined preset, or create a new one.
        self.select_default_colors.setHidden(True)
        self.select_your_own_colors.setHidden(True)
        self.define_colors.setHidden(True)
        

        self.folder_list.setHidden(False)
        self.folder_name.setHidden(False)
        self.select_energy.setHidden(True)
        self.select_momentum.setHidden(True)

        self.select_acceleration.setHidden(False)
        self.select_acceleration.isChecked()

    def skeletal_tracker_pressed(self):
        self.set_folder_to_other()
        type_of_tracking = 'other'
        input_folder = 'other_i'
        data_output = 'other_o' 
        
        print('skeletal')
        webbrowser.open('https://physicsrule.github.io/SkeletalTracking.github.io/')
        #webbrowser.get("google-chrome").open('https://physicsrule.github.io/SkeletalTracking.github.io/')
    
    def load_data(self,type_of_tracking, input_folder, title_of_table):
        # load the dictionary
        ## TODO make this a file to read
        objects_to_track = [{
       'color' : "green" , 'lower' : (29, 67, 6) , 'upper' : (64, 255, 255) , 'radius' : 10 , 'mass' : 0.0},
        {'color' : "red" , 'lower' : (0, 146, 12) , 'upper' : (11, 255, 206) , 'radius' : 10 , 'mass' : 0.0},
        {'color' : "blue" , 'lower' : (58, 71, 52) , 'upper' : (125, 255, 170) , 'radius' : 10 , 'mass' : 0.0},
        {'color' : "purple" , 'lower' : (139,  68,  78) , 'upper' : (170, 255, 255) , 'radius' : 10 , 'mass' : 0.0},
        {'color' : "yellow" , 'lower' : (20, 36, 4) , 'upper' : (71, 238, 213), 'radius' : 10 , 'mass' : 0.0},
        {'color' : "orange" , 'lower' : (0, 123, 189), 'upper' : (24, 255, 255) , 'radius' : 10 , 'mass' : 0.0}]
        row = 0
        title_of_table.setRowCount(len(objects_to_track))
        for object in objects_to_track:
            # Checkbox to chose default colors
            item = QTableWidgetItem(''.format(row, 0))
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Unchecked)
            title_of_table.setItem(row, 0, item)
            # Columns for default colors
            title_of_table.setItem(row,1, QTableWidgetItem(object[type_of_tracking]))
            title_of_table.setItem(row,2, QTableWidgetItem(str(object['radius'])))
            title_of_table.setItem(row,3, QTableWidgetItem(str(object['mass'])))
            title_of_table.setItem(row,4, QTableWidgetItem(str((object['lower']))))
            title_of_table.setItem(row,5, QTableWidgetItem(str((object['upper']))))
            item_on_screen = QTableWidgetItem(''.format(row, 6))
            item_on_screen.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled)
            item_on_screen.setCheckState(Qt.CheckState.Unchecked)
            title_of_table.setItem(row, 6, item_on_screen )
            row +=1

    def default_colors_shown(self):
        type_of_tracking ='color'
        input_folder = 'color_i'
        data_output = 'color_o'
        # Set Data table for the colors to track
        self.table_widget_color.setColumnWidth(0,50)
        self.table_widget_color.setColumnWidth(1,100)
        self.table_widget_color.setColumnWidth(2,100)
        self.table_widget_color.setColumnWidth(3,100)
        self.table_widget_color.setColumnWidth(4,0)
        self.table_widget_color.setColumnWidth(5,0)
        self.table_widget_color.setColumnWidth(6,0)
        objects_to_track  = self.load_data(type_of_tracking, input_folder,self.table_widget_color) 
        self.table_widget_color.setHidden(False)
        self.table_widget_color_2.setHidden(True)
        self.lineEdit_define_color_name.setHidden(True)
        self.combo_box_objects.setHidden(True)
        # saved colors
        self.folder_name_objects.setHidden(True)
        self.find_lower_upper_button.setHidden(True)

    def your_own_colors_shown(self):
        stype_of_tracking ='color'
        input_folder = 'color_i'
        data_output = 'color_o'
        # default colors
        self.table_widget_color.setHidden(True)
        self.combo_box_objects.setHidden(False)
        # saved colors
        self.folder_name_objects.setHidden(False)
        self.table_widget_color_2.setHidden(True)
        self.lineEdit_define_color_name.setHidden(True)
        self.find_lower_upper_button.setHidden(True)
        
        self.select_object_file(input_folder)

    def define_colors_shown(self):
        type_of_tracking ='color'
        input_folder = 'color_i'
        data_output = 'color_o'
        # default colors
        self.table_widget_color.setHidden(True)
        self.combo_box_objects.setHidden(True)
        self.folder_name_objects.setHidden(True)

        self.table_widget_color_2.setHidden(False)
        self.lineEdit_define_color_name.setHidden(False)

        self.table_widget_color_2.setColumnWidth(0,30)
        self.table_widget_color_2.setColumnWidth(1,50)
        self.table_widget_color_2.setColumnWidth(2,63)
        self.table_widget_color_2.setColumnWidth(3,52)
        self.table_widget_color_2.setColumnWidth(4,80)
        self.table_widget_color_2.setColumnWidth(5,80)
        objects_to_track  = self.load_data(type_of_tracking, input_folder, self.table_widget_color_2) 
        self.table_widget_color_2.setColumnWidth(6,56)
        self.find_lower_upper_button.setHidden(False)


    def color_button_pressed(self):
        self.data_output = 'color_o'
        # Select the color from a list, use a predefined preset, or create a new one.
        self.select_default_colors.setHidden(False)
        self.select_your_own_colors.setHidden(False)
        self.define_colors.setHidden(False)

        self.folder_list.setHidden(False)
        self.folder_name.setHidden(False)
        self.select_acceleration.setHidden(False)
        self.select_momentum.setHidden(False)
        self.select_energy.setHidden(True)
        '''
        self.select_energy.setHidden(False)
        '''

        # Folder List Showing Folders
        self.list_folders(self.data_output)

    # Color Selected
    def color_selected(self):
        # Define the the folders that will be used      
        self.folder_list.setHidden(False)
        self.folder_name.setHidden(False)
        self.select_acceleration.setHidden(False)
        self.select_momentum.setHidden(False)
        '''
        self.select_energy.setHidden(False)
        '''
        self.select_energy.setHidden(True)
        # Folder List Showing Folders
        data_output = 'color_o'
        self.list_folders(data_output)
        
    

    def find_trendlines_button_pressed(self):
        base_path = os.getcwd()
        # trendline_table_widget has variables required stored starting at row 20
        calc_file_name_path = GUI_graph_trendline(self.trendline_table_widget, self.graph_widget)
        self.graph_widget.draw()
        GUI_show_equations_on_table(self.trendline_table_widget, calc_file_name_path )
          

# Presets appear
    def show_presets(self):
        print('We do not have presets option ready yet. Coming soon')
        pass

    # ft Preset Selected
    ## TODO once you select a preset it creates the array
    def ft_presets(self):
        pass

        # Pass through function
    def get_output_folder_path(self, base_path, data_output):
        # output directory to store the data
        dir_path = os.path.abspath(os.path.join(base_path, 'data', data_output, ''))
        data_output_folder = self.folder_name.text()
        if self.folder_name.text() == '':
            data_output_folder = 'default'
            print('Select the folder to save into')
            ## TODO erase existing folders in default
        data_output_folder_path = os.path.abspath(os.path.join(dir_path, data_output_folder, ''))

        return data_output_folder_path


    # Choice of Acceleration, Momentum, or Energy
    # Hides the variables that momentum does not need so it cleans up the visual if used energy etc.

    def acceleration_chosen(self):
        self.up_axis.setHidden(True)
        self.r_height.setHidden(True)
        self.save_height_mass.setHidden(True)
        self.mass_of_object.setHidden(True)

    def momentum_chosen(self):
        self.up_axis.setHidden(True)
        self.r_height.setHidden(True)
        self.save_height_mass.setHidden(True)
        ''' TODO when adding graphing of any Time,x,y,z
        self.mass_of_object.setHidden(False)
        '''

    def energy_chosen(self):
        self.up_axis.setHidden(False)
        self.r_height.setHidden(False)
        self.save_height_mass.setHidden(False)
        self.mass_of_object.setHidden(False)

    def reference_height_save(self):
        try:
            height_display = float(self.r_height.text())
        except ValueError:
            print('The reference height was not saved. Reference height of 0 meters was used')
            height_display = 0.0
    # Get all necessary settings to start tracking 
    def get_settings(self) -> Tuple[int, str, image_option, List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str, float, float]], int, str]:
        ## TODO How do I put all of this in a function so I can call some of it from each tracker?
                # Select video feed camera
        base_path = os.getcwd()
        
        # Set this to True if you are testing
        if False:
            src = 4
            type_of_tracking = 'color'
            self.image = image_option(True,False,False,False,False,False,False)
            # This is the green color range
            self.color_ranges=np.array([([32, 70, 68], [ 64, 194, 227], 'green', 0.1, 0.)],dtype=np.dtype([('lower', np.int32, (3,)), ('upper', np.int32, (3,)), ('name', np.unicode_, 16), ('radius_meters', np.float32), ('mass', np.float32)]))
            min_radius_object=5
            data_output_folder_path=self.get_output_folder_path(base_path, '169')

            # when tracking
            # data_output_folder_path=self.get_output_folder_path(base_path, str(time.time()))
            input_folder='color_i'
            data_output ='color_o'
            return src, type_of_tracking, self.image, self.color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output

        #  self.image = get_settings_to_pass(self)
        self.image = image_option(self.show_image_check.isChecked(), self.save_image_check.isChecked(), self.show_depth_check.isChecked(), self.save_depth_check.isChecked(), self.show_tracking_mask_check.isChecked(),self.save_tracking_mask_check.isChecked(), self.save_video.isChecked())
        
        ## TODO Add spot in GUI for this later. How do I do this?
        min_radius_object = 5

        # pass the folder names for correct storage location and type of tracking
        ## pass type of tracking into this function
        type_of_tracking ='color'
        if type_of_tracking == 'color':
            input_folder = 'color_i'
            data_output = 'color_o'
        elif type_of_tracking == 'infrared':
            input_folder = 'infrared_i'
            data_output = 'infrared_o'
        
        data_output_folder_path = self.get_output_folder_path(base_path, data_output)

        # Store the array of colors to track in the output directory
        # color choice
        dt = np.dtype([('lower', np.int32, (3,)),('upper', np.int32, (3,)), ('name', np.unicode_, 16), ('radius_meters', np.float32),('mass', np.float32)])

        num_objects = 0
        # object count
        i = 0
        new_color_ranges =[]
        if type_of_tracking == 'color':
            if self.select_default_colors.isChecked() or self.define_colors.isChecked():
                if self.select_default_colors.isChecked():
                # Just use the default colors and adjust the mass and radius
                    title_of_table = self.table_widget_color
                    self.color_ranges_text = "default_colors"
                elif self.define_colors.isChecked():
                # You have modified your own colors and color names to suit your situation. 
                # They will be saved and you can name them
                    title_of_table = self.table_widget_color_2
                
                # Read the colors from one of the 2 tables above
                name_of_array = ''
                for row in range(title_of_table.rowCount()):
                    if title_of_table.item(row,0).checkState() == Qt.CheckState.Checked:
                        item = QTableWidgetItem(''.format(row, 1))
                        color = title_of_table.item(row,1).text()
                        print('color', color)
                        lower = title_of_table.item(row,4).text()
                        upper = title_of_table.item(row,5).text()
                        radius_meters = float(title_of_table.item(row,2).text())/100
                        mass = title_of_table.item(row,3).text()

                        the_array = np.array([(ast.literal_eval(lower),ast.literal_eval(upper), (color),(radius_meters), (mass) )],dtype=dt)
                        if i==0:
                            new_color_ranges = the_array
                        else:
                            new_color_ranges = np.hstack((new_color_ranges,the_array))
                        i += 1
                    self.color_ranges = new_color_ranges
             
                if self.define_colors.isChecked():
                    self.color_ranges_text = self.lineEdit_define_color_name.text()
                    dir_path = os.path.abspath(os.path.join(base_path, 'data', input_folder, ''))
                    dir_path_npy= os.path.abspath(os.path.join(dir_path, self.color_ranges_text,''))
                    np.save(dir_path_npy, new_color_ranges)

            elif self.select_your_own_colors.isChecked():
            # You have previously defined the colors
                self.color_ranges_text = self.combo_box_objects.currentText()
                dir_path = os.path.abspath(os.path.join(base_path, 'data', input_folder, ''))
                dir_path_npy= os.path.abspath(os.path.join(dir_path, self.color_ranges_text,''))
                if self.select_your_own_colors.isChecked() :
                    self.color_ranges = np.load(dir_path_npy)
                    print('color range from previous own colors' , self.color_ranges)                 
            

            ##TODO save the numpy as a text file as well so it is easy to read
            #txt_npy_name = str(self.color_ranges_text + '.txt')
            #dir_out_path_txt_npy = os.path.abspath(os.path.join(data_output_folder_path , txt_npy_name )) 
            #np.savetxt(dir_out_path_txt_npy, np.array(self.color_ranges))

        # Object tracking. These values put in so we can use same tracker program as color    
        elif type_of_tracking == 'obj_tracker':
            lower = " "
            upper = " "
            ## TODO if you want mulitple objects, have a hstack of the number of objects to track
            ## TODO XXX have the user input the radius !!!
            radius_meters = 0.10
            ##TODO Have the user input the mass
            mass = 0.0
            self.color_ranges = np.array([(ast.literal_eval(lower),ast.literal_eval(upper), (color),(radius_meters), (mass) )],dtype=dt)
            self.color_ranges_text = 'nothing'

        # Save the np file in the output folder so you know which one was used
        dir_out_path_npy = os.path.abspath(os.path.join(data_output_folder_path , self.color_ranges_text,'')) 
        np.save(dir_out_path_npy , self.color_ranges)
        ## TODO use a different value if needed to use src
        src=4

        # data_output_folder, data_output_folder_path = make_new_folder(data_output)
        return src, type_of_tracking, self.image, self.color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output


    def setup_trendline_table(self, title_of_table,csv_files_array, data_output, folder_name, data_output_folder_path, trendline_folder_path):
        parameters = parameter_to_plot(self.select_acceleration.isChecked(), self.select_momentum.isChecked(), self.select_energy.isChecked())
        self.trendline_table_widget.setHidden(False)
        self.find_trendlines_button.setHidden(False)
        title_of_table.setColumnCount(len(csv_files_array))
        title_of_table.setRowCount(30)
        
        title_of_table.setVerticalHeaderItem(0, QTableWidgetItem('object name'))
        title_of_table.setVerticalHeaderItem(1, QTableWidgetItem('mass (kg)'))
        title_of_table.setVerticalHeaderItem(2, QTableWidgetItem('Minimum time'))
        title_of_table.setVerticalHeaderItem(3, QTableWidgetItem('Maximum time'))
        
        column = 0
        for (__, file_name, mass) in csv_files_array:
            title_of_table.setItem(0,column, QTableWidgetItem(file_name))
            title_of_table.setItem(1,column, QTableWidgetItem(str(mass)))
            
            for i,var in enumerate(['x','y','z']):
                row = i + 4
                title_of_table.setVerticalHeaderItem(row, QTableWidgetItem( str((var)+' function') ))
                function_type_combo_box = QComboBox()
                function_type_combo_box.addItems(['linear', 'quadratic', 'future',])
                title_of_table.setCellWidget(row, column, function_type_combo_box)
                # set default value for combo_box
                # function_type_combo_box.SelectedValue = "linear"
            # Next object
            column += 1 

        row = 7
        for i,var in enumerate(['x','y','z']):
            var_of_t = str(str(var)+'(t)') 
            v_var_of_t = str('V'+ str(var)+'(t)') 
            if parameters.which_parameter_to_plot =='E':
                if i == 0:      third_var_of_t = 'PE(t)'
                elif i == 1:    third_var_of_t = 'KE(t)'
                else:           third_var_of_t = 'Total(t)'
            else:
                third_var_of_t = str(parameters.which_parameter_to_plot + str(var_of_t)) # Example Ax or Py
            print(i)
            title_of_table.setVerticalHeaderItem(row + i*4, QTableWidgetItem(str('R^2 for ' + var_of_t)))
            title_of_table.setVerticalHeaderItem(row + i*4 + 1, QTableWidgetItem(var_of_t))
            title_of_table.setVerticalHeaderItem(row + i*4 + 2, QTableWidgetItem(v_var_of_t))
            title_of_table.setVerticalHeaderItem(row + i*4 + 3, QTableWidgetItem(third_var_of_t))
        
        column = 0
        # pass information to the trendline procedure in the table so user can see too
        title_of_table.setVerticalHeaderItem(20,QTableWidgetItem( 'data_output' ))
        title_of_table.setItem(20,column,QTableWidgetItem( str(data_output) ))
        title_of_table.setVerticalHeaderItem(21,QTableWidgetItem( 'folder_name' ))
        title_of_table.setItem(21,column,QTableWidgetItem( str(folder_name) ))
        title_of_table.setVerticalHeaderItem(22,QTableWidgetItem( 'data_output_folder_path' ))
        title_of_table.setItem(22,column,QTableWidgetItem( str(data_output_folder_path) ))
        title_of_table.setVerticalHeaderItem(23,QTableWidgetItem( 'csv_files_array' ))
        title_of_table.setItem(23,column,QTableWidgetItem( str(csv_files_array) ))
        title_of_table.setVerticalHeaderItem(24,QTableWidgetItem( 'parameter_to_plot' ))    
        title_of_table.setItem(24,column,QTableWidgetItem( parameters.which_parameter_to_plot ))
        title_of_table.setVerticalHeaderItem(25,QTableWidgetItem( 'trendline_folder_path' ))
        title_of_table.setItem(25,column,QTableWidgetItem( str(trendline_folder_path) ))

        


    def run_graph(self, data_output_folder_path):
        
        # The folder that will be graphed
        print('graphing')
        # What variable is to be graphed for the 3rd graph. It always graphs position and velocity
        if self.select_momentum.isChecked(): which_parameter_to_plot = 'p'
        elif self.select_energy.isChecked(): which_parameter_to_plot = 'e'
        # acceleration is default
        else : which_parameter_to_plot = 'a'

        if self.data_in_other.isChecked(): 
            data_output = 'color_o'
            self.select_momentum.isHidden(True)
            self.select_energy.isHidden(True)
            which_parameter_to_plot = 'a'
        else:
            data_output = 'color_o'
        base_path = os.getcwd()

        data_output_folder_path = self.get_output_folder_path(base_path, data_output)
        
        # The array of the colors to be tracked
        graph_color_ranges, csv_files_array = find_objects_to_graph (data_output_folder_path)

        # self.grid_layout.addWidget(self.graph_widget_3D, 0, 1, alignment=Qt.Alignment())

        line_style_array, line_color_array, marker_shape_array, show_legend = plot_style_color()
        self.graph_widget.clear()
        self.graph_widget, points_to_smooth = GUI_graph_setup(self.graph_widget, which_parameter_to_plot)
        trendline_folder_path, smooth_data_to_graph = GUI_graph (which_parameter_to_plot, data_output_folder_path, graph_color_ranges, csv_files_array, points_to_smooth )

        i_object=0
               
        for (file_path, file_name, mass) in csv_files_array:
            file_name_dataframe = file_name  + "sheet.csv"
            file_name_dataframe_path = os.path.abspath(os.path.join(trendline_folder_path + '/' + file_name_dataframe + '/' ))     
            smooth_data_to_graph = pd.read_csv(file_name_dataframe_path, header=0)
            data_frame = pd.DataFrame(smooth_data_to_graph) 
            smooth_data_to_graph = data_frame.set_index('Time')

            #plot data
            for i,var in enumerate(['x','y','z']):
                smooth_data_to_graph[var].plot(ax=self.graph_widget.axes[0,i], title=var, fontsize=8, marker=marker_shape_array[i_object], markersize=3, color=line_color_array[i_object],  linestyle=line_style_array[i_object], label=file_name)
                v_var = str('V'+ str(var)) 
                smooth_data_to_graph[v_var].plot(ax=self.graph_widget.axes[1,i], title=v_var, fontsize=8, marker=marker_shape_array[i_object], markersize=3, color=line_color_array[i_object], linestyle=line_style_array[i_object])
                if which_parameter_to_plot in {"a","A"}:
                    a_var = str('A'+ str(var)) 
                    smooth_data_to_graph[a_var].plot(ax=self.graph_widget.axes[2,i], title=a_var, fontsize=8, marker=marker_shape_array[i_object], markersize=3, color=line_color_array[i_object], linestyle=line_style_array[i_object])
                elif which_parameter_to_plot in {"p","P"}:
                    ##if 'CM' not in filename:
                    p_var = str('P'+ str(var)) 
                    smooth_data_to_graph[p_var].plot(ax=self.graph_widget.axes[2,i], title=p_var, fontsize=8, marker=marker_shape_array[i_object], markersize=3, color=line_color_array[i_object], linestyle=line_style_array[i_object])
            if which_parameter_to_plot =='E':
                smooth_data_to_graph['KE'].plot(ax=self.graph_widget.axes[2,0], title='KE', fontsize=8, marker=marker_shape_array[i_object], markersize=3, color=line_color_array[i_object], linestyle=line_style_array[i_object])        
                smooth_data_to_graph['PE'].plot(ax=self.graph_widget.axes[2,1], title='PE', fontsize=8, marker=marker_shape_array[i_object], markersize=3, color=line_color_array[i_object], linestyle=line_style_array[i_object])
                smooth_data_to_graph['Total'].plot(ax=self.graph_widget.axes[2,2], title='Total E', fontsize=8, marker=marker_shape_array[i_object], markersize=3, color=line_color_array[i_object], linestyle=line_style_array[i_object])
        
            i_object +=1

        for count_i in range(3):
            for count_j in range(3):
                self.graph_widget.axes[count_i,count_j].minorticks_on()   
                if count_i==2: 
                    self.graph_widget.axes[count_i,count_j].set(xlabel='Time (s)') 

        if show_legend == "y":
            self.graph_widget.axes[0, 0].legend(loc="upper right", shadow=True, fancybox=True, fontsize=8)

        self.graph_widget.draw()
        self.Button3DGraph.setHidden(False)
        self.setup_trendline_table(self.trendline_table_widget, csv_files_array, data_output, self.folder_name.text(), data_output_folder_path, trendline_folder_path)
        

        #plt.tight_layout()

    def run_3D_graph(self, data_output_folder_path):
        self.xmin, self.xmax = self.graph_widget.axes[0,0].get_xlim()
        fig_3D = plt.figure()
        axes_3D = plt.axes(projection='3d')

        axes_3D.set(zlabel='y')     # Switched the y and z as most of the time the camera has y vertical as it is set on a table facing outward
        axes_3D.set_xlabel('x (meters)', labelpad=25)
        axes_3D.set_ylabel('z (meters)', labelpad=25) 
        axes_3D.set_zlabel('y (meters)', labelpad=20)
        line_style_array, line_color_array, marker_shape_array, show_legend = plot_style_color()
        

        print (self.xmin, self.xmax)

        # The folder that will be graphed
        base_path = os.getcwd()
        data_output = 'color_o'
        data_output_folder_path = self.get_output_folder_path(base_path, data_output)
        
        # The array of the colors to be tracked
        graph_color_ranges, csv_files_array = find_objects_to_graph (data_output_folder_path)

        # Column header on position data 
        header_list = ['Time', 'x', 'y', 'z']
        # i represents each object tracked
        i=0
        i_object =0
        for (file_path, file_name, mass) in csv_files_array:
        # For each object create a data_file from position data

            # Setup data_frame to put velocity, momentum, acceleration, and energy data
            file_name_dataframe = file_name  + "sheet.csv"
            file_name_dataframe_path = os.path.abspath(os.path.join(data_output_folder_path + '/' + file_name_dataframe + '/' ))   

            # Reads the current version of the *csv file and smooths it
            file_name_w_extension = file_name + '.csv'
            path_to_file = os.path.abspath(os.path.join(file_path, file_name_w_extension))
            graph_data = pd.read_csv(path_to_file, header=0, names = header_list)
            
            # Dont need data_frame = pd.DataFrame(graph_data) 
            # Dont need smooth_data_to_graph = data_frame.set_index('Time')
            
            # Trim
            graph_data_window = graph_data[graph_data["Time"]>self.xmin]
            graph_data_window = graph_data_window[graph_data_window["Time"]<self.xmax]
            
            axes_3D.plot3D(graph_data_window['x'],graph_data_window['z'], graph_data_window['y'], line_color_array[i_object], linestyle= line_style_array[i_object], markersize = 3, marker = marker_shape_array[i_object])
            axes_3D.scatter3D(graph_data_window['x'],graph_data_window['z'], graph_data_window['y'], c=line_color_array[i_object], cmap=line_color_array[i_object], marker = marker_shape_array[i_object])
            i_object +=1
        plt.ioff()    
        plt.show()
        self.graph_widget.draw()
        
    def find_lower_upper_bounds(self):
        # print(self.table_widget_color_2.rowCount())
        self.table_widget_color_2 = reload_table(self.table_widget_color_2 )
        self.color_ranges_text = self.lineEdit_define_color_name.text



    # Run Tracker Button Function
    def run_real_time(self):
        print('running real-time')
        self.DataGraph.setHidden(True)
        src, type_of_tracking, image, color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output = self.get_settings()
        GUI_real_time_color_tracking(src, type_of_tracking, image ,color_ranges , min_radius_object, data_output_folder_path)

    def run_tracker(self):
        src, type_of_tracking, image, color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output = self.get_settings()
        # config by looking at the camera (remove this from tracking program below)
        # find output folder here instead of 
        pipeline = find_and_config_device()
        GUI_tracking(pipeline, src, type_of_tracking, image, color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output)
        

    def record_bag(self):
        print('recording video')
        src, type_of_tracking, image, color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output = self.get_settings()
        record_bag_file(data_output_folder_path, type_of_tracking)

    def track_from_bag(self):
        print('Tracking from a previously recorded bag file.')
        src, type_of_tracking, image, color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output = self.get_settings()
        #config by opening pipeline from bag)
        types_of_streams_saved = 'color' #or 'infrared' at 90 fps or 'id300' at 300 fps
        data_output_folder = self.folder_name.text()
        if self.folder_name.text() == '':
            data_output_folder = 'default'
            print('Select the folder to save into')
            ## TODO erase existing folders in default
        bag_file = 'bag.bag'
        bag_folder_path =  os.path.abspath(os.path.join(data_output_folder_path + "/" + bag_file))
        pipeline = read_bag_file_and_config(types_of_streams_saved, data_output_folder_path, data_output_folder , bag_folder_path)

        GUI_tracking(pipeline, src, type_of_tracking, image, color_ranges, min_radius_object, data_output_folder_path, input_folder, data_output)



    def toggle_window(self, window):
        if window.isVisible():
            window.hide()

        else:
            window.show()
class Worker1(QThread):
    ImageUpdate = pyqtSignal()
    def run(self):
        self.ThreadActive = True
        # Possible Cameras
        index = 0
        arr = []
        src_count = 6
        while index < src_count > 0:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.read()[0]:
                ret, frame = cap.read()
                img_name = "cam_img_{}.png".format(index)
                cv2.imwrite(img_name, frame)
                cap.release()
                print('camera #',index)
            index += 1

        self.ImageUpdate.emit()
    def stop(self):
        self.ThreadActive = False
        self.quit()

def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()


if __name__ == '__main__':
    main()
