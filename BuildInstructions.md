# Build Instructions
The information below provides instructions on how to build opencv with opencv_contrib which is required to execute the GMS demo application and for the resulting HGMS project. The information provided below is for Windows based machines.

## Dependencies
1. Install CMake (https://cmake.org/). This will also come bundled with Visual Studio 2022 community edition though may be an older version
2. Install Visual Studio 2022 Community edition (https://visualstudio.microsoft.com/vs/community/) C++ support. The build can also be done by through vs.code but c++ compiler dependencies will need to be installed
3. Download this repo with opencv and opencv_contrib

## Generate Build Config
1. In your local repo at the same level as the opencv and opencv_contrib folders, create two folders
    a. build -> this folder will contain the build config and related source/solution files
    b. opencv_install  -> this folder will contain the compile lib and dll files
2. To generate the build config you can use the CMAKE app or the "Developer Command Prompt for Visual Studio 2022"
3. If using the command prompt, then change directories to your new build folder and enter the following cmake command:
```
   cmake ^
  -DOPENCV_EXTRA_MODULES_PATH=*_path to your opencv_contrib/modules folder_* ^
  -DCMAKE_INSTALL_PREFIX=*_path to your opencv_install folder_* ^
  -DWITH_CUDA=OFF ^
  -DBUILD_opencv_xfeatures2d=ON ^
  -DBUILD_opencv_world=ON ^
  -DCMAKE_BUILD_TYPE=Debug ^
  ../opencv
```
5. The commands in step 3 require specifying the path to opencv_contrib/modules and opencv_install folder. You can try relative paths, but I found absolute path (**E.g. c:\source\587\opencv_contrib\modules**) to work better for me. The command instruct opencv to build with extra modules, turn off cuda support, combine all built lib's/dll's install single world files, and make the debug version of the resulting files.
6. After the config is complete, the run the following cmake command to build the source and generate the lib/dll files in the opencv_install folder:
```
cmake --build . --config Debug --target INSTALL
```
6. The build will take a while and then the resulting files will be available in opencv_install. If you have a version of opencv installed on your computer and in your user/system path then this reference will need to be removed to avoid dll conflicts. You can add the newly generated dll to your user/system path or add the generate dll file to your source folder.

