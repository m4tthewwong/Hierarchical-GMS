//---------------------------------------------------------------------------  
// MPStage.h
// Multi-plane stage header file for representing multi-plane GMS processing stage. 
// This class derives from ProcessingStage and provides implementation for multi-plane
// GMS feature match filtering. This class supports the following operations:
//  1. 
// Authors:  Matthew Wong, Brennan O’Reilly, Pranshu Bhardwaj
//---------------------------------------------------------------------------
// Inputs:
//  -- 
// 
// Outputs:
// -- 
// 
// Description:
//    This class provides the Multi-plane stage implementation.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
// 

#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "ProcessingStage.h"

class MPStage : public ProcessingStage {
public:
    /*----------------------------- default -----------------------------------
    * Default constructor for class MPStage.
    * Preconditions: none
    * Postconditions: Instance of MPStage class is instantiated
    */
    MPStage();

    /*----------------------------- destructor -------------------------------
    * Destructor for class MPStage.
    * Preconditions: Instance of MPStage class is instantiated.
    * Postconditions: Objects are deallocated.
    */
    virtual ~MPStage();

    /*----------------------------- execute -------------------------------
    * Execute method for class MPStage.
    * Preconditions: Instance of MPStage class is instantiated and image 1
    *                and 2 keypoints are provided, image1 and 2 sizes are provided,
    *                image 1 and 2 feature matches are provided, thresholdFactor
    *                is provided.
    * Postconditions: The execute method is run and algorithm executed
    */
    virtual void execute(const std::vector<cv::KeyPoint>& vkp1, const cv::Size& size1,
        const std::vector<cv::KeyPoint>& vkp2, const cv::Size& size2,
        const std::vector<cv::DMatch>& matchesAll, std::vector<cv::DMatch>& vDMatches,
        const double thresholdFactor);

private:
    // class identifier 
    static constexpr const char* STAGE_NAME = "MP";

    // ---------------------- helper functions ----------------------

    static std::vector<cv::Point2f> computeMotionVectors(
        const std::vector<cv::KeyPoint>& vkp1,
        const std::vector<cv::KeyPoint>& vkp2,
        const std::vector<cv::DMatch>& matches);

    static std::vector<int> clusterMotionVectors(
        const std::vector<cv::Point2f>& motion,
        const int K);

    static void filterPlaneMatches(
        const std::vector<cv::DMatch>& planeMatches,
        const std::vector<cv::KeyPoint>& vkp1,
        const std::vector<cv::KeyPoint>& vkp2,
        const double thresholdFactor,
        std::vector<cv::DMatch>& inliersOut);
};
