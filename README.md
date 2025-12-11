# Hierarchical-GMS
The purpose of this repo is to provide source code for the final project in UW Bothell CSS587. 

## Folder Structure
The folder structure is as follows:
- *Hgms/Hgms:* Source code for HGMS project including pipeline code, HGMS Stage, LAT Stage, MP Stage, Homography evaluator for evaluating performance, and sample image files for testing with HGMSDemo.cpp.
- *Paper Presentation Demo:* Includes GmsDemo that was included when presenting the original GMS paper which served as inspiration for HGMS project.
- *VGG Test Results:* Test results from Hgms/Hgms source code and includes csv files capturing baseline, GMS, HGMS performance for the multi-stage processing pipeline in filter and aggregation modes and for each pipeline stage.
- *opencv:* OpenCV source 4.12.0 -> OpenCV and OpenCV_Contrib are required for building of these projects.
- *opencv_contrib:* OpenCV_Contrib 4.12.0 
To build the opencv/opencv_contrib source code please refer to OpenCV_BuildInstructions.md.

## HGMS Demo
The HGMSDemo.cpp file contains the "main" method which can run the following:
- Pipeline unit tests
- VGG image benchmark tests for performance comparison against GMS (requires separate download and configuration of VGG Affine Covariant image data set: https://www.robots.ox.ac.uk/~vgg/research/affine/
- Demo

The behavior of the "main" method is controller through 3 boolean flags **(RUN_UNIT_TESTS, RUN_BENCHMARK_TESTS, RUN_PIPELINE_DEMO)**. If these flags are set to true, then the "main" method will run corresponding methods for each flag. Other parameters worth noting are the **SOURCE_IMAGE** and **DEST_IMAGE** parameters. These parameters control which images are used for the demo. The demo also contains a live video portion which requires camera access on the computer running the demo.

## Solution Build
The files in the Hgms/Hgms folder (cpp, h, and hpp) can be added to any **empty** Visual Studio 2022 project (other versions may work but this was only verified using Visual Studio 2022 community edition. To successfully built you must configure the properties of the project with a property sheet file that contains the location of OpenCV/OpenCV_Contrib includes and lib files. Additionally the **opencv_world4120d.dll** file with OpenCV_Contrib xfeatures2d module must either be on the system path or located within the same folder as the compiled HGMSDemo executable. If the **opencv_world4120d.dll** is not found then the HGMSDemo executable will build but will fail to run.

## HGMS Example
<img width="2387" height="892" alt="lambo-hgms" src="https://github.com/user-attachments/assets/ed41363a-de18-4b85-8e0e-6d9f745a0e98" />


