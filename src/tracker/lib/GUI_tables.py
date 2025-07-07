"""Library of functions that Load Tables into the GUI
The default colors
The user defines colors to to put into the table
The objects that don't have colors used for object tracking
"""

import ast
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import Qt
import numpy as np

from tracker.lib.GUI_tracker import find_lower_upper_bounds_on_screen

def load_ranges(title_of_table):
    # Load a table that was created
    dt = np.dtype([('lower', np.int32, (3,)),('upper', np.int32, (3,)), ('name', np.str_, 16), ('radius_meters', np.float32),('mass', np.float32)])
    i = 0
    name_of_array = ''
    for row in range(title_of_table.rowCount()):
        if title_of_table.item(row,0).checkState() == Qt.CheckState.Checked:
            item = QTableWidgetItem(''.format(row, 1))
            color =         title_of_table.item(row,1).text()
            radius_meters = float(title_of_table.item(row,2).text())/100
            mass =          float(title_of_table.item(row,3).text())
            lower =         title_of_table.item(row,4).text()
            upper =         title_of_table.item(row,5).text()
                     
            the_array = np.array([(ast.literal_eval(lower),ast.literal_eval(upper), (color),(radius_meters), (mass) )],dtype=dt)
            if i==0:
                new_color_ranges = the_array
            else:
                new_color_ranges = np.hstack((new_color_ranges,the_array))
            i += 1
    return new_color_ranges

def load_object_ranges(title_of_table):
    # Load a table that was created from objects
    ## TODO make all table loading and reading json
    dt_obj = np.dtype([('lower', np.str_, 16),('upper', np.str_, 16), ('name', np.str_, 16), ('radius_meters', np.float32),('mass', np.float32)])
    # note, upper and lower not used, but could be for background subtraction
    i = 0
    for row in range(title_of_table.rowCount()):
        if title_of_table.item(row,0).checkState() == Qt.CheckState.Checked:
            item = QTableWidgetItem(''.format(row, 1))
            color =         title_of_table.item(row,1).text()
            radius_meters = float(title_of_table.item(row,2).text())/100
            mass =          float(title_of_table.item(row,3).text())
            lower =         ' '
            upper =         ' '
                     
            the_array = np.array([((lower),(upper), (color),(radius_meters), (mass) )],dtype=dt_obj)
            if i==0:
                new_color_ranges = the_array
            else:
                new_color_ranges = np.hstack((new_color_ranges,the_array))
            i += 1
    return new_color_ranges

def load_data(type_of_tracking, input_folder, title_of_table):
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


def reload_table (title_of_table):
# after adjusting the upper and lower bounds modify the table
    dt = np.dtype([('lower', np.int32, (3,)),('upper', np.int32, (3,)), ('name', np.str_, 16), ('radius_meters', np.float32),('mass', np.float32)])
    for row in range(title_of_table.rowCount()):
        if title_of_table.item(row,6).checkState() == Qt.CheckState.Checked:
            item = QTableWidgetItem(''.format(row, 1))
            color = title_of_table.item(row,1).text()
            lower = title_of_table.item(row,4).text()
            upper = title_of_table.item(row,5).text()
            print('Changing...', color)
            radius_meters = float(title_of_table.item(row,2).text())/100
            mass = title_of_table.item(row,3).text()
            the_array = np.array([(ast.literal_eval(lower),ast.literal_eval(upper), (color),(radius_meters), (mass) )],dtype=dt)
            the_array = find_lower_upper_bounds_on_screen(the_array)
            [(lower, upper, color, radius_meters, mass)] = the_array
            title_of_table.setItem(row,4, QTableWidgetItem(str(lower)))
            title_of_table.setItem(row,5, QTableWidgetItem(str(upper)))
    return title_of_table


def load_data_objects(title_of_table):
    # load the dictionary
    ## TODO make this a file to read
    objects_to_track = [{
    'name' : "object1" ,  'radius' : 10 , 'mass' : 0.0},
    {'name' : "object2" , 'radius' : 10 , 'mass' : 0.0},
    {'name' : "object3" , 'radius' : 10 , 'mass' : 0.0},
    {'name' : "object4" , 'radius' : 10 , 'mass' : 0.0},
    {'name' : "object5" , 'radius' : 10 , 'mass' : 0.0},
    {'name' : "object6" , 'radius' : 10 , 'mass' : 0.0}]
    row = 0
    title_of_table.setRowCount(len(objects_to_track))
    for object in objects_to_track:
        # Checkbox to chose default colors
        item = QTableWidgetItem(''.format(row, 0))
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled)
        if row == 0: 
            item.setCheckState(Qt.CheckState.Checked)
        else: 
            item.setCheckState(Qt.CheckState.Unchecked)
        title_of_table.setItem(row, 0, item)
        # Columns for default colors
        title_of_table.setItem(row,1, QTableWidgetItem(str(object['name'])))
        #title_of_table.setItem(row,2, QTableWidgetItem(str(object['lower'])))
        #title_of_table.setItem(row,3, QTableWidgetItem(str(object['upper'])))
        title_of_table.setItem(row,2, QTableWidgetItem(str(object['radius'])))
        title_of_table.setItem(row,3, QTableWidgetItem(str(object['mass'])))
        title_of_table.setItem(row,4, QTableWidgetItem("(0, 0, 0)"))    # Lower bounds from color
        title_of_table.setItem(row,5, QTableWidgetItem("(0, 0, 0)"))    # upper bounds from color
        # For future selection of object??
        item_on_screen = QTableWidgetItem(''.format(row, 6))
        item_on_screen.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled)
        item_on_screen.setCheckState(Qt.CheckState.Unchecked)
        title_of_table.setItem(row, 4, item_on_screen )
        row +=1

