# functions for the GUI to keep the main file cleaner
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import *
from pyqtgraph.Qt import QtGui, QtCore
import ast

import numpy as np
from tracker.lib.GUI_color_tracker import find_lower_upper_bounds_on_screen


def reload_table (title_of_table):
# after adjusting the upper and lower bounds modify the table
    dt = np.dtype([('lower', np.int32, (3,)),('upper', np.int32, (3,)), ('name', np.unicode_, 16), ('radius_meters', np.float32),('mass', np.float32)])
    print(title_of_table)
    for row in range(title_of_table.rowCount()):
        if title_of_table.item(row,6).checkState() == Qt.CheckState.Checked:
            item = QTableWidgetItem(''.format(row, 1))
            color = title_of_table.item(row,1).text()
            lower = title_of_table.item(row,4).text()
            upper = title_of_table.item(row,5).text()
            print('hi', color, lower, upper)
            radius_meters = float(title_of_table.item(row,2).text())/100
            mass = title_of_table.item(row,3).text()
            the_array = np.array([(ast.literal_eval(lower),ast.literal_eval(upper), (color),(radius_meters), (mass) )],dtype=dt)
            the_array = find_lower_upper_bounds_on_screen(the_array)
            print(the_array ,'\n')
            [(lower, upper, color, radius_meters, mass)] = the_array
            title_of_table.setItem(row,4, QTableWidgetItem(str(lower)))
            title_of_table.setItem(row,5, QTableWidgetItem(str(upper)))
    return title_of_table

