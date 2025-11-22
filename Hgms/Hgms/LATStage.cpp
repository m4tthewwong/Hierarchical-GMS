//---------------------------------------------------------------------------  
// LATStage.cpp
// Local affine transformation stage header file for representing local affine GMS 
// processing stage. This class derives from ProcessingStage and provides 
// implementation for local affine GMS feature match filtering. This class 
// supports the following operations:
//	1. Perform local affine (RANSAC-based) refinement on grid-based matches
//	STEP 1. Create Soure and Destination grids
//  STEP 2. Discretize feature matches from source into count matrix
//  STEP 3. sum up row counts and store as last column in count matrix
//  STEP 4: Identify rows with high enough sums (e.g. matches), run RANSAC to create matrix
//  STEP 5: Apply matrix to cell center to derive center of target cell
//  STEP 6: Return filtered inliers
// Authors:  Brennan O'Reilly, Pranshu Bhardwaj, Matthew Wong
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

/*----------------------------- getCellIndex -------------------------------
* Private method to return index of cell in grid
* Preconditions: Instance of LATStage class is instantiated and image 1
*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
*                image 1 and 2 feature matches are provided, thresholdFactor
*                is provided.
* Postconditions: Returns index of cell or returns -1 if the point is outside
*                 the image bounds.
*/
int LATStage::getCellIndex(const cv::Point2f& pt,
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

	const float cellWidth = static_cast<float>(size.width) / static_cast<float>(gridCols);
	const float cellHeight = static_cast<float>(size.height) / static_cast<float>(gridRows);

	int col = static_cast<int>(pt.x / cellWidth);
	int row = static_cast<int>(pt.y / cellHeight);

	// Clamp for safety
	col = std::max(0, std::min(gridCols - 1, col));
	row = std::max(0, std::min(gridRows - 1, row));

	return row * gridCols + col;
}

/*----------------------------- calcCellCenter -------------------------------
* Private method to calculate the cell center
* Preconditions:
* Postconditions:
*/
Point2f LATStage::calcCellCenter(int linearIndex, 
	const Size& size, 
	int gridRows, 
	int gridCols) const
{
	// Determine the row and column index from the linear index
	int row = linearIndex / gridCols;
	int col = linearIndex % gridCols;

	// Calculate dimensions of a single cell
	float cellWidth = static_cast<float>(size.width) / static_cast<float>(gridCols);
	float cellHeight = static_cast<float>(size.height) / static_cast<float>(gridRows);

	// Calculate the center coordinates: (Col index + 0.5) * Cell Width
	float x = (col + 0.5f) * cellWidth;
	float y = (row + 0.5f) * cellHeight;

	return Point2f(x, y);
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

	// Const number of cells for grids
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

	//countingMatrix 400 x 400 to represent counting stats and source grid x target grid
	std::vector<int> countingMatrix(numCells * (numCells), 0);

	//retrieve matches and add counts of source/target matches into countingMatrix
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

	// -----------------------------------------------------------------
	// Step 3: RANSAC - create storage for affine models
	// Identify the rows with the highest count of features
	// For each row with the highest count (past threshold) translate the row back into
	//  the source cell index in the 20 x 20 source grid.
	// Retrieve feature matches (vector start and end) from source and target
	// Run RANSAC over features matches to generate affine matrix (2 x 3 matrix)
	// ----------------------------------------------------------------
	std::vector<Mat> srcAffineMatrix(numCells);
	std::vector<int> finalScores(numCells, 0);

	// iterate through source matches and if count > match threshold then create matrix
	for (int i = 0; i < numCells; i++)
	{
		if (matchesBySrcCell[i].size() > RANSAC_MATCH_THRESHOLD)
		{
			// convert matches to point correspondences for estimateAffine2D
			std::vector<Point2f> pts1, pts2;
			pts1.reserve(matchesBySrcCell[i].size());
			pts2.reserve(matchesBySrcCell[i].size());

			// retrieve matches that exist in source cell
			// for each set of matches in cell, get src and dst points
			for (int matchIdx : matchesBySrcCell[i])
			{
				const DMatch& match = matchesAll[matchIdx];
				pts1.push_back(vkp1[match.queryIdx].pt);
				pts2.push_back(vkp2[match.trainIdx].pt);
			}

			// run RANSAC on src, dst points (at least 3 points needed for RANSAC)
			Mat affineMatrix, inliers;
			affineMatrix = estimateAffine2D(pts1, pts2, inliers, cv::RANSAC, 3.0);

			// Save model to run on cell centers
			if (!affineMatrix.empty())
			{
				srcAffineMatrix[i] = affineMatrix;
			}
		}
	}

	// STEP 4: Apply matrix to cell center to derive center of target cell
	// Take affine and apply to cell center to derive prediction of where
	// matches should be transformed in the target image
	// Convert from count matrix cell back to source grid cell index
	// Higher scores indicates matches found where matches were predicted by the
	//  affine transform
	for (int i = 0; i < numCells; i++)
	{
		Mat affine = srcAffineMatrix[i];

		if (!affine.empty())
		{
			// determine center coordinates for cell with matrix
			Point2f cellCenter = calcCellCenter(i, size1, gridRows, gridCols);

			std::vector<Point2f> srcCenter = { cellCenter };
			std::vector<Point2f> predCenter;
			
			// apply matrix to srcCenter to predict target cell center
			cv::transform(srcCenter, predCenter, affine);
			
			// find predicted target cell index to score matches
			const int predTgCellIdx = getCellIndex(predCenter[0], size2, gridRows, gridCols);

			// based on predicted target cell, counts for neighborhood cells 3 x 3
			// must also be retrieved to determine final score
			// convert from counting matrix to image cell
			int predCellRow = predTgCellIdx / gridCols;
			int predCellCol = predTgCellIdx % gridCols;

			// Retrieve and sum counts from 3 x 3 neighborhood
			int neighSumScore = 0;
			for (int neighCellIdx = 0; neighCellIdx < 9; neighCellIdx++)
			{

				// get the neighborhood cell locations
				int neighCol = predCellCol + NEIGHBORHOOD_CALC_OFFSETS[neighCellIdx][0];
				int neighRow = predCellRow + NEIGHBORHOOD_CALC_OFFSETS[neighCellIdx][1];

				// check that col and row are within bounds for counting
				if (neighRow >= 0 && neighRow < gridRows &&
					neighCol >= 0 && neighCol < gridCols)
				{
					// convert neighCol and neighRow back to countMatrix index
					int neighTgtCell = neighRow * gridCols + neighCol;

					// retrieve score and add to total neighSumScore
					neighSumScore += countingMatrix[i * numCells + neighTgtCell];
				}
			}

			// add neighborhood total sum score to the final scores for the center cell
			finalScores[i] = neighSumScore;
		}
	}

	// -------------------------------------------------------------------
	// Step 6: Filter Matches that are greater than score threshold and return as inliers
	// -------------------------------------------------------------------
	vDMatches.reserve(matchesAll.size());

	for (int i = 0; i < numCells; i++)
	{
		// check if score is greater than final score threshold
		if (finalScores[i] >= FINAL_SCORE_THRESHOLD * thresholdFactor)
		{
			for (int matchIndex : matchesBySrcCell[i])
			{
				vDMatches.push_back(matchesAll[matchIndex]);
			}
		}
	}
}