# Packaging python into standalone executable
(When reading this document, replace MY_PROGRAM with the name of your program)

<!-- if '../LICENSE:.' doesn't work use '../LICENSE:.\' -->
## To create a executible folder with the program and data files
1. Open terminal in vscode and switch to bash
2. In bash type `pip install pyinstaller`
3. In bash type `pyinstaller --specpath build/ --contents-directory "." --add-data '..\src\tracker\lib\GUI_Base.ui;.\src\tracker\lib' --add-data '..\LICENSE;.' --add-data '../data;.\data' -D src/GUI_main_program.py`
4. Copy the dist into a folder on a flash drive or c:drive with a main folder named `3D`
5. Make sure the data folder has the following folders, if not add: color_i, color_o, infrared_i, infrared_o, other_i, other_o
6. Check the dist folder for a file called `MY_PROGRAM.exe`

## Running the Executable
1. Double click the *.exe to reun the prgram
2. You DO NOT need to download python. It is included in the dist folder.
3. The program will take a minute to start the first time.

### Data directory
In your current directory a folder named `data` MUST exist. 
You need the following folders in the data folder: color_i, color_o, infrared_i, infrared_o, other_i, other_o

### If there are errors
Double clicking works, but if you need to see an error:
- open a terminal in the project root (the directory this file is in)
    - if you're using command prompt (If your prompt looks like `C:\Users\Bob>`) run `.\dist\MY_PROGRAM.exe`
    - if you're using bash (if our prompt looks like `$ `) run `./dist/MY_PROGRAM.exe`