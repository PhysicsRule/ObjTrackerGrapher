These programs track an object with the Intel RealSense D435i in 3D, saves it to a csv file. It will also graph and find trendlines of the objects position, velocity, and acceleration vs time.

Videos on how to use the program: https://drive.google.com/drive/folders/1jAxUofq5oxr5pNKwH6UzhtLq5SLXYAgR?usp=sharing

# To setup the program to make modifications:
Have the folder near your c: drive or on a flashdrive
## Install Python and SDK for the Camera
1. Install Python 3.7 
2. Install Intel RealSense SKD 2.0 and pyrealsense2 so you can run python projects with the camera
## Install Libraries
To avoid having to add all of the python libraries individually, add the requirements.txt.
 ## Download the data folder
1. Download the data.zip folder into the main folder of your program. There are some sample data files to work with and the program requires them to run
2. Ignore the data folder by selecting the data folder: cntl-shift-p add gitignore
### If using Visual Studio...:
  1. Add the python environment 3.7
  2. Right click on the python 3.7 environment and install from requirements.txt for each program or do it with a virtual environment.(Ctrl+Shift+P), start typing the Python: Create Environment command to search, and then select the command.
 
 # To download the *.exe and associated files to quickly run off of a flash drive:
 1. Download python (version 2 if you might want to mofidy the code) 
    a. Check : Add python in the PATH
    b. Disable path length limit
 2. in VSCode: 
    a. Download Python extension
      i. Create virtual environment : `venv`
      ii. restart the terminal
  
 Read Compiling.md
