# Packaging python into standalone executable
(When reading this document, replace MY_PROGRAM with the name of your program)

<!-- if '../LICENSE:.' doesn't work use '../LICENSE:.\' -->
```bash
pip install pyinstaller
1. pyinstaller --specpath build/ --add-data '..\src\tracker\lib\GUI_Base.ui;.\src\tracker\lib' --add-data '..\LICENSE;.' --add-data '../data;.\data' -D src/GUI_main_program.py

2. copy the dist into a folder on a flash drive with a main folder 3D
3. make sure the data folder has the following folders, if not add: color_i, color_o, infrared_i, infrared_o, other_i, other_o


For example when trying to run 

## Requirements dir
- tracker MUST be in the same directory as the .py program

## Output
Check the dist folder for a file called `MY_PROGRAM.exe`

## Running the Executable

### Data directory
In your current directory a folder named `data` MUST exist. 
You need the following folders in the data folder: color_i, color_o, infrared_i, infrared_o, other_i, other_o


Double clicking works, but if you need to see an error:
- open a terminal in the project root (the directory this file is in)
    - if you're using command prompt (If your prompt looks like `C:\Users\Bob>`) run `.\dist\MY_PROGRAM.exe`
    - if you're using bash (if our prompt looks like `$ `) run `./dist/MY_PROGRAM.exe`