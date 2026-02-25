- After unpacking, extract minecraft.jar assets into projects's assets folder.
- Python 3.14 or higher may be required.
 "Pillow" python module may also have to be installed for python script to work.
 Try running it before installing, all required packages may be included.
 Can be done in command prompt, running "pip3 install pillow".
 Check online for more installation info.


- Once installed, run "start.bat" from root directory.
- Set your "Steamapps" folder location.
- Select one or multiple minecraft json models in the Model JSON field. Point to your project's assets folder.
- From there, its assets -> minecraft -> models -> block, then select one or multiple files.
  Select your game from dropdown.
- Select "make models" for the default mode, "model scale" will be used, default 48.
- Select "make skybox models" for models scaled by the Skybox Scale, default 16.
- Click "Convert" to start conversion.
- QC, QCI and SMD files can be found in "assets/mcexports/" folder.
- All models compiled and textures will be added to the game folder in models and materials under minecraft.


- Only Minecraft models are currently supported in the version.
  Some models may not work like chest since the json file doesn't contain a parent file pointing to
  the elements required to build the model. Other json files my error for unknown reasons.
 
