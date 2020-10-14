# Packaging python into standalone executable
(When reading this document, replace MY_PROGRAM with the name of your program)

```bash
pip install pyinstaller
1. pyinstaller --specpath build/ --add-data '../src/tracker/lib/GUI_Base.ui:.\src\tracker\lib' -D src/GUI_Tracker.py

2. copy the dist into a folder on a flash drive with a main folder 3D
3. copy the data folder with color_i, color_o and some example files into the drive
4. copy the videos onto the drive as well outside the 3D folder

For example when trying to run 

## Requirements dir
- tracker MUST be in the same directory as the .py program

## Output
Check the dist folder for a file called `MY_PROGRAM.exe`

## Running the Executable

### Data directory
In your current directory a folder named `data` MUST exist. 
You may need color_i and color_o folders as well.

Double clicking works, but if you need to see an error:
- open a terminal in the project root (the directory this file is in)
    - if you're using command prompt (If your prompt looks like `C:\Users\Bob>`) run `.\dist\MY_PROGRAM.exe`
    - if you're using bash (if our prompt looks like `$ `) run `./dist/MY_PROGRAM.exe`