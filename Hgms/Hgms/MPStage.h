//---------------------------------------------------------------------------  
// MPStage.h
// Multi-plane stage header file for representing multi-plane GMS processing stage. 
// This class derives from ProcessingStage and provides implementation for multi-plane
// GMS feature match filtering. This class supports the following operations:
//  1. Segment feature matches into motion-based planes using motion vector clustering  
//  2. Perform lightweight GMS-like consistency filtering within each plane  
//  3. Aggregate filtered inliers from all planes into a final refined match set
// 
// Authors:  Matthew Wong, Brennan O’Reilly, Pranshu Bhardwaj
//---------------------------------------------------------------------------
// Inputs:
//  -- Keypoints for image 1 and image 2  
//  -- Sizes of image 1 and image 2  
//  -- Full match list (matchesAll) produced by previous stages  
//  -- thresholdFactor for adjusting filtering strictness  
// 
// Outputs:
//  -- vDMatches: filtered inlier match set after multi-plane refinement  
// 
// Description:
//    This class provides the Multi-plane (MP) stage implementation, which is
//    designed to improve match robustness in scenes containing parallax or
//    multiple depth layers. The stage clusters feature match motion vectors
//    into K motion planes and then applies a simplified GMS-like consistency
//    test independently within each plane. This allows matches belonging to
//    different depth layers or independently moving regions to be preserved
//    rather than incorrectly filtered out by a single global model.
// 
//    The MP stage is typically applied after the initial HGMS stage has
//    produced a coarse but robust match set, and after the LAT stage has
//    optionally refined matches locally. MP then separates these refined
//    matches into motion-consistent groups and filters each group separately.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
//   -- Scenes may contain multiple depth layers or parallax motion  
//   -- A small fixed number of planes (K=2) is used unless adapted later
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

    /*----------------------------- computeMotionVectors -------------------------------
    * Helper function to compute motion vectors (p2 - p1) for each feature match.
    * Preconditions: vkp1, vkp2, and matches contain valid indexed keypoints/matches.
    * Postconditions: Returns a vector of 2D motion vectors corresponding to matches.
    */
    static std::vector<cv::Point2f> computeMotionVectors(
        const std::vector<cv::KeyPoint>& vkp1,
        const std::vector<cv::KeyPoint>& vkp2,
        const std::vector<cv::DMatch>& matches);

    /*----------------------------- clusterMotionVectors -------------------------------
    * Helper function to cluster motion vectors into K motion planes using k-means.
    * Preconditions: motion contains at least one valid motion vector; K >= 1.
    * Postconditions: Returns a label array assigning each motion vector to a plane.
    */
    static std::vector<int> clusterMotionVectors(
        const std::vector<cv::Point2f>& motion,
        const int K);

    /*----------------------------- filterPlaneMatches -------------------------------
    * Helper function to apply a GMS-like filtering step within a single motion plane.
    * Preconditions: planeMatches contains valid matches belonging to the same plane.
    * Postconditions: inliersOut contains matches consistent with plane motion statistics.
    */
    static void filterPlaneMatches(
        const std::vector<cv::DMatch>& planeMatches,
        const std::vector<cv::KeyPoint>& vkp1,
        const std::vector<cv::KeyPoint>& vkp2,
        const double thresholdFactor,
        std::vector<cv::DMatch>& inliersOut);
};
