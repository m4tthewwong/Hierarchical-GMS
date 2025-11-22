//---------------------------------------------------------------------------  
// LATStage.cpp
// Local affine transformation stage header file for representing local affine GMS 
// processing stage. This class derives from ProcessingStage and provides 
// implementation for local affine GMS feature match filtering. This class 
// supports the following operations:
//	1. Perform local affine (RANSAC-based) refinement on grid-based matches
// Authors:  Brennan O�Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- Keypoints for image1 and image2, image sizes, initial matches, thresholdFactor
// 
// Outputs:
// -- Filtered inlier matches after local affine refinement
// 
// Description:
//    This class provides the local affine transformation stage implementation.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
//  

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>

#include "LATStage.h"

// Anonymous namespace for internal helper functions (not exposed in header)
namespace
{
	// -----------------------------------------------
	// Helper: compute grid cell index for a point
	// -----------------------------------------------
	// Returns -1 if the point is outside the image bounds.
	inline int getCellIndex(const cv::Point2f& pt,
	                        const cv::Size& size,
	                        const int gridRows,
	                        const int gridCols)
	{
		if (pt.x < 0.f || pt.y < 0.f ||
		    pt.x >= static_cast<float>(size.width) ||
		    pt.y >= static_cast<float>(size.height))
		{
			return -1;
		}

		const float cellWidth  = static_cast<float>(size.width)  / static_cast<float>(gridCols);
		const float cellHeight = static_cast<float>(size.height) / static_cast<float>(gridRows);

		int col = static_cast<int>(pt.x / cellWidth);
		int row = static_cast<int>(pt.y / cellHeight);

		// Clamp for safety
		col = std::max(0, std::min(gridCols - 1, col));
		row = std::max(0, std::min(gridRows - 1, row));

		return row * gridCols + col;
	}
}

/*----------------------------- default -----------------------------------
* Default constructor for class LATStage.
* Preconditions: Name of stage is required input parameter
* Postconditions: Instance of LATStage class is instantiated
*/
LATStage::LATStage() : ProcessingStage(STAGE_NAME)
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class LATStage.
* Preconditions: Instance of LATStage class is instantiated.
* Postconditions: Objects are deallocated.
*/
LATStage::~LATStage()
{
}

/*----------------------------- execute -------------------------------
* Execute method for class LATStage.
* Preconditions: Instance of LATStage class is instantiated and image 1
*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
*                image 1 and 2 feature matches are provided, thresholdFactor
*                is provided.
* Postconditions: The execute method is run and algorithm executed
*/
void LATStage::execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
	const std::vector<KeyPoint>& vkp2, const Size& size2,
	std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
	const double thresholdFactor)
{
    // If there are no matches or insufficient keypoints, nothing to do
	if (matchesAll.empty() || vkp1.empty() || vkp2.empty())
	{
		return;
	}

	// Clear output for this stage; we will repopulate vDMatches.
	vDMatches.clear();

	// Grid parameters (Step 1: Create source and destination 20 x 20 grids)
	const int gridRows = 20;
	const int gridCols = 20;
	const int numCells = gridRows * gridCols;

    // ---------------------------------------------------------------------
	// Step 1:
	// Create source and destination 20 x 20 grids
	// Represented implicitly via getCellIndex() helper; we also build
	// a mapping from source-cell -> list of match indices.
	// ---------------------------------------------------------------------
	std::vector<std::vector<int>> matchesBySrcCell(numCells);
	std::vector<int> countingMatrix(numCells * numCells, 0); // 400 x 400 flattened

	for (int i = 0; i < static_cast<int>(matchesAll.size()); ++i)
	{
		const DMatch& m = matchesAll[i];

		// Safety checks on indices
		if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(vkp1.size()) ||
		    m.trainIdx < 0 || m.trainIdx >= static_cast<int>(vkp2.size()))
		{
			continue;
		}

		const Point2f& p1 = vkp1[m.queryIdx].pt;
		const Point2f& p2 = vkp2[m.trainIdx].pt;

		const int srcCell = getCellIndex(p1, size1, gridRows, gridCols);
		const int dstCell = getCellIndex(p2, size2, gridRows, gridCols);

		if (srcCell < 0 || dstCell < 0)
		{
			continue;
		}

        // -----------------------------------------------------------------
		// Step 2:
		// Discretize the feature matches from source into counting matches
		// 400 x 400 array (source cell x destination cell).
		// -----------------------------------------------------------------
		countingMatrix[srcCell * numCells + dstCell] += 1;

		// Track which matches belong to which source cell
		matchesBySrcCell[srcCell].push_back(i);
	}

    // ---- QUICK TEST ----
    int nonZero = 0;
    for (int i = 0; i < countingMatrix.size(); i++) {
        if (countingMatrix[i] != 0) nonZero++;
    }
    std::cout << "[LAT TEST] Non-zero entries in countingMatrix: " << nonZero << std::endl;
    
    for (int c = 0; c < matchesBySrcCell.size(); c++) {
        if (!matchesBySrcCell[c].empty()) {
            std::cout << "[LAT TEST] Cell " << c
                    << " has " << matchesBySrcCell[c].size()
                    << " matches." << std::endl;
            break; // only print first non-empty cell to keep output small
        }
    }

}