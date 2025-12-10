//---------------------------------------------------------------------------  
// LATStage.h
// Local affine transformation stage header file for representing local affine GMS 
// processing stage. This class derives from ProcessingStage and provides 
// implementation for local affine GMS feature match filtering. This class 
// supports the following operations:
//	1. Perform local affine (RANSAC-based) refinement on grid-based matches
//	STEP 1. Create Soure and Destination grids
//  STEP 2. Discretize feature matches from source into count matrix
//  STEP 3: Identify rows with high enough sums (e.g. matches), run RANSAC to create matrix
//  STEP 4: Apply matrix to cell center to derive center of target cell
//  STEP 5: Return filtered inliers
// Authors:   Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- Keypoints for image 1 and image 2  
//  -- Sizes of image 1 and image 2  
//  -- Full match list (matchesAll) produced by previous stages  
//  -- thresholdFactor for adjusting filtering strictness  
// 
// // Outputs:
// -- vDMatches: filtered inlier match set after local affine transform refinement 
// 
// Description:
//    This class provides the local affine transformation stage implementation, 
//    which is designed to predict how neighborhood clusters will transform from
//    source to target image and then filter feature matches based on RANSAC prediction.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
// 

#pragma once

#include <string>
#include "ProcessingStage.h"

class LATStage : public ProcessingStage {
public:
	/*----------------------------- default -----------------------------------
	* Default constructor for class LATStage.
	* Preconditions: none
	* Postconditions: Instance of LATStage class is instantiated
	*/
	LATStage();

	/*----------------------------- destructor -------------------------------
	* Destructor for class LATStage.
	* Preconditions: Instance of LATStage class is instantiated.
	* Postconditions: Objects are deallocated.
	*/
	virtual ~LATStage();

	/*----------------------------- execute -------------------------------
	* Execute method for class LATStage.
	* Preconditions: Instance of LATStage class is instantiated and image 1
	*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
	*                image 1 and 2 feature matches are provided, thresholdFactor
	*                is provided.
	* Postconditions: The execute method is run and algorithm executed
	*/
	virtual void execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
		const std::vector<KeyPoint>& vkp2, const Size& size2,
		const std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
		const double thresholdFactor);

private:
	// Const class identifier 
	static constexpr const char* STAGE_NAME = "LAT";

	// Threshold for determining number of matches before performing RANSAC
	const int RANSAC_MATCH_THRESHOLD = 5;

	// Threshold for number of counts that must exist in target cell to keep match
	const int FINAL_SCORE_THRESHOLD = 1;

	// This 2d array stores the offsets to retrieve the neighborhood counts
	// based on the predicted target center.
	// The predicted target center is at (0, 0) and the center of a 3 x 3 array
	const int NEIGHBORHOOD_CALC_OFFSETS[9][2] = {
		{-1, -1}, {0, -1}, {1, -1},
		{-1, 0}, {0, 0}, {1, 0},
		{-1, 1}, {0, 1}, {1, 1} };


	/*----------------------------- getCellIndex -------------------------------
	* Private method to return index of cell in grid
	* Preconditions: Instance of LATStage class is instantiated and image 1
	*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
	*                image 1 and 2 feature matches are provided, thresholdFactor
	*                is provided.
	* Postconditions: Returns index of cell or returns -1 if the point is outside 
	*                 the image bounds.
	*/
	int getCellIndex(const cv::Point2f& pt,
		const cv::Size& size,
		const int gridRows,
		const int gridCols);

	/*----------------------------- calcCellCenter -------------------------------
	* Private method to calculate the cell center
	* Preconditions: Valid cell index, size, number of rows/cols in grid
	* Postconditions: Point2f calculation of cell center
	*/
	Point2f calcCellCenter(int linearIndex, 
		const Size& size, 
		int gridRows, 
		int gridCols) const;
};
