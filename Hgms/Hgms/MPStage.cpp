//---------------------------------------------------------------------------  
// MPStage.cpp
// Multi-plane stage header file for representing multi-plane GMS processing stage. 
// This class derives from ProcessingStage and provide implement for multi-plane
// GMS feature match filtering. This class supports the following operations:
//  1. Segment matches into motion-based image planes
//  2. For each plane, perform a lightweight GMS-like consistency filtering
//  3. Aggregate inlier matches from all planes
// Authors:  Matthew Wong, Brennan O�Reilly, Pranshu Bhardwaj
//---------------------------------------------------------------------------
// Inputs:
//  -- Keypoints for image1 and image2, image sizes, and current match set
//     (matchesAll)
// 
// Outputs:
// -- vDMatches: filtered inlier matches after multi-plane refinement
// 
// Description:
//    Stage 3 provides support for parallax images or scenes with background
//    and foreground objects at different depths. The stage clusters feature
//    matches into motion planes and then runs a simple GMS-like consistency
//    check per plane. The resulting inliers from all planes are merged into
//    a single set of matches.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
//   -- HGMS and LAT stages have already been applied if necessary and provided a 
//      reasonably good initial set of matches in matchesAll.
//---------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core.hpp>

#include "MPStage.h"

using cv::Point2f;
using cv::Mat;
using cv::TermCriteria;

// Anonymous namespace for internal helper functions
namespace
{
    /*----------------------------- computeMotionVectors ----------------------
	* Helper for Step 1:
	* Compute motion vectors (p2 - p1) for each match.
	*/
	std::vector<Point2f> computeMotionVectors(
		const std::vector<cv::KeyPoint>& vkp1,
		const std::vector<cv::KeyPoint>& vkp2,
		const std::vector<cv::DMatch>& matches)
	{
		std::vector<Point2f> motion;
		motion.reserve(matches.size());

		for (const auto& m : matches)
		{
			// Safety checks on indices
			if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(vkp1.size()) ||
				m.trainIdx < 0 || m.trainIdx >= static_cast<int>(vkp2.size()))
			{
				continue;
			}

			Point2f p1 = vkp1[m.queryIdx].pt;
			Point2f p2 = vkp2[m.trainIdx].pt;
			motion.emplace_back(p2 - p1);
		}

		return motion;
	}

	/*----------------------------- clusterMotionVectors ----------------------
	* Helper for Step 1:
	* Cluster motion vectors into K planes using k-means.
	* Returns a label (0..K-1) for each motion vector.
	*/
	std::vector<int> clusterMotionVectors(
		const std::vector<Point2f>& motion,
		const int K)
	{
		std::vector<int> labels;

		if (motion.empty() || K <= 1)
		{
			// Single plane (all zeros) if not enough data or K <= 1
			labels.assign(motion.size(), 0);
			return labels;
		}

		const int N = static_cast<int>(motion.size());
		Mat data(N, 2, CV_32F);

		for (int i = 0; i < N; ++i)
		{
			data.at<float>(i, 0) = motion[i].x;
			data.at<float>(i, 1) = motion[i].y;
		}

		Mat bestLabels;
		Mat centers;

		cv::kmeans(
			data,
			K,
			bestLabels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.01),
			3,
			cv::KMEANS_PP_CENTERS,
			centers
		);

		labels.resize(N);
		for (int i = 0; i < N; ++i)
		{
			labels[i] = bestLabels.at<int>(i, 0);
		}

		return labels;
	}
}

/*----------------------------- default -----------------------------------
* Default constructor for class MPStage.
* Preconditions: Name of stage is required input parameter
* Postconditions: Instance of MPStage class is instantiated
*/
MPStage::MPStage() : ProcessingStage(STAGE_NAME)
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class MPStage.
* Preconditions: Instance of MPStage class is instantiated.
* Postconditions: Objects are deallocated.
*/
MPStage::~MPStage()
{
}

/*----------------------------- execute -------------------------------
* Execute method for class MPStage.
* Preconditions: Instance of MPStage class is instantiated and image 1
*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
*                image 1 and 2 feature matches are provided, thresholdFactor
*                is provided.
* Postconditions: The execute method is run and algorithm executed
*/
void MPStage::execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
	const std::vector<KeyPoint>& vkp2, const Size& size2,
	std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
	const double thresholdFactor)
{
    // Silence unused warnings for size1/size2
	(void)size1;
	(void)size2;

	// Clear output for this stage
	vDMatches.clear();

	// If there are no matches or insufficient keypoints, nothing to do
	if (matchesAll.empty() || vkp1.empty() || vkp2.empty())
	{
		return;
	}

    // ---------------------------------------------------------------------
	// Step 1: Segment image planes
	//   - Compute motion vectors for each match
	//   - Cluster motion vectors into K planes
	// ---------------------------------------------------------------------
	std::vector<Point2f> motion = computeMotionVectors(vkp1, vkp2, matchesAll);

	// If after safety checks we lost most matches, just pass through
	if (motion.size() < 3)
	{
		vDMatches = matchesAll;
		return;
	}

	// I have chosen a small fixed number of planes (K). This can be made adaptive.
    // To explain this more:
    // 1 plane, if there is no parallax
    // 2 planes, if there is foreground and background
    // 3+ planes, if there are objects at several depths
	const int K = 2;
	std::vector<int> labels = clusterMotionVectors(motion, K);

	// Count number of non-empty planes
	std::vector<int> planeCounts(K, 0);
	for (int lbl : labels)
	{
		if (lbl >= 0 && lbl < K)
		{
			planeCounts[lbl]++;
		}
	}

	int nonZeroPlanes = 0;
	for (int c : planeCounts)
	{
		if (c > 0) ++nonZeroPlanes;
	}

	// If effectively only a single plane, pass matches through
	if (nonZeroPlanes <= 1)
	{
		vDMatches = matchesAll;
		return;
	}

    // ---------------------------------------------------------------------
	// Step 2: For each plane, collect matches and run local filtering
	// ---------------------------------------------------------------------

    // ---------------------------------------------------------------------
	// Step 3: Aggregate inlier matches from all planes and return
	// ---------------------------------------------------------------------
}