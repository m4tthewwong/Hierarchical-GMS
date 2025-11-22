//---------------------------------------------------------------------------  
// LATStage.h
// Local affine transformation stage header file for representing local affine GMS 
// processing stage. This class derives from ProcessingStage and provides 
// implementation for local affine GMS feature match filtering. This class 
// supports the following operations:
//	1. 
// Authors:   Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- 
// 
// Outputs:
// -- 
// 
// Description:
//    This class provides the local affine transformation stage implementation.
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
		std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
		const double thresholdFactor);

private:
	// Const class identifier 
	static constexpr const char* STAGE_NAME = "LAT";

	// Threshold for determining number of matches before performing RANSAC
	const int RANSAC_MATCH_THRESHOLD = 50;
	
	// Threshold for number of counts that must exist in target cell to keep match
	const int FINAL_SCORE_THRESHOLD = 5;

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
	* Preconditions: 
	* Postconditions:
	*/	Point2f calcCellCenter(int linearIndex, 
		const Size& size, 
		int gridRows, 
		int gridCols) const;

};
