# Hierarchical-GMS
The purpose of this repo is to provide source code for the final project in UW Bothell CSS587. 

The folder structure is as follows:
- *Hgms/Hgms:* Source code for HGMS project including pipeline code, HGMS Stage, LAT Stage, MP Stage, Homography evaluator for evaluating performance, and sample image files for testing with HGMSDemo.cpp.
- *Paper Presentation Demo:* Includes GmsDemo that was included when presenting the original GMS paper which served as inspiration for HGMS project.
- *VGG Test Results:* Test results from Hgms/Hgms source code and includes csv files capturing baseline, GMS, HGMS performance for the multi-stage processing pipeline in filter and aggregation modes and for each pipeline stage.
- *opencv:* OpenCV source 4.12.0 -> OpenCV and OpenCV_Contrib are required for building of these projects.
- *opencv_contrib:* OpenCV_Contrib 4.12.0 
To build the opencv/opencv_contrib source code please refer to OpenCV_BuildInstructions.md.
