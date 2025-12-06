//---------------------------------------------------------------------------  
// MPStage.cpp
// Multi-plane stage implementation file for representing the multi-plane GMS 
// processing stage. This class derives from ProcessingStage and provides the 
// concrete implementation for motion-plane segmentation and intra-plane match 
// refinement. This stage supports the following operations:
//  1. Segment feature matches into motion-based planes using motion vector clustering  
//  2. Perform lightweight GMS-like consistency filtering within each plane  
//  3. Aggregate filtered inliers from all planes into a final refined match set
// Authors:  Matthew Wong, Brennan O'Reilly, Pranshu Bhardwaj
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
//---------------------------------------------------------------------------

#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core.hpp>

#include "MPStage.h"

using cv::Point2f;
using cv::Mat;
using cv::TermCriteria;


// ==========================================================================
//  PRIVATE HELPER FUNCTIONS
// ==========================================================================

/*----------------------------- computeMotionVectors -------------------------------
* Helper function to compute motion vectors (p2 - p1) for each feature match.
* Preconditions: vkp1, vkp2, and matches contain valid indexed keypoints/matches.
* Postconditions: Returns a vector of 2D motion vectors corresponding to matches.
*/
std::vector<Point2f> MPStage::computeMotionVectors(
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



/*----------------------------- clusterMotionVectors -------------------------------
* Helper function to cluster motion vectors into K motion planes using k-means.
* Preconditions: motion contains at least one valid motion vector; K >= 1.
* Postconditions: Returns a label array assigning each motion vector to a plane.
*/
std::vector<int> MPStage::clusterMotionVectors(
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



/*----------------------------- filterPlaneMatches -------------------------------
* Helper function to apply a GMS-like filtering step within a single motion plane.
* Preconditions: planeMatches contains valid matches belonging to the same plane.
* Postconditions: inliersOut contains matches consistent with plane motion statistics.
*/
void MPStage::filterPlaneMatches(
    const std::vector<cv::DMatch>& planeMatches,
    const std::vector<cv::KeyPoint>& vkp1,
    const std::vector<cv::KeyPoint>& vkp2,
    const double thresholdFactor,
    std::vector<cv::DMatch>& inliersOut)
{
    if (planeMatches.size() < 3)
    {
        // Not enough data for meaningful statistics; pass through
        inliersOut.insert(inliersOut.end(), planeMatches.begin(), planeMatches.end());
        return;
    }

    std::vector<Point2f> motions;
    motions.reserve(planeMatches.size());

    for (const auto& m : planeMatches)
    {
        if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(vkp1.size()) ||
            m.trainIdx < 0 || m.trainIdx >= static_cast<int>(vkp2.size()))
        {
            continue;
        }

        Point2f p1 = vkp1[m.queryIdx].pt;
        Point2f p2 = vkp2[m.trainIdx].pt;
        motions.emplace_back(p2 - p1);
    }

    if (motions.size() < 3)
    {
        inliersOut.insert(inliersOut.end(), planeMatches.begin(), planeMatches.end());
        return;
    }

    // Compute mean motion
    Point2f mean(0.f, 0.f);
    for (const auto& mv : motions)
    {
        mean.x += mv.x;
        mean.y += mv.y;
    }
    mean.x /= static_cast<float>(motions.size());
    mean.y /= static_cast<float>(motions.size());

    // Compute variance of motion distance from mean
    float variance = 0.f;
    for (const auto& mv : motions)
    {
        const float dx = mv.x - mean.x;
        const float dy = mv.y - mean.y;
        variance += dx * dx + dy * dy;
    }
    variance /= static_cast<float>(motions.size());
    const float stddev = std::sqrt(std::max(variance, 0.0f));

    // Here I did thresholdFactor / 6.0 because I believe it should be a sort of middle ground
    // where < 6 where would be looser filtering and > 6 would be stricter filtering.
    const float baseFactor = 1.5f;
    const float scale = static_cast<float>(thresholdFactor > 0.0 ? (thresholdFactor / 6.0) : 1.0);
    const float stddevFactor = baseFactor * scale;
    const float thresh = stddevFactor * stddev;

    // Apply consistency check
    inliersOut.reserve(inliersOut.size() + planeMatches.size());

    for (size_t i = 0; i < planeMatches.size(); ++i)
    {
        const cv::DMatch& m = planeMatches[i];

        if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(vkp1.size()) ||
            m.trainIdx < 0 || m.trainIdx >= static_cast<int>(vkp2.size()))
        {
            continue;
        }

        Point2f p1 = vkp1[m.queryIdx].pt;
        Point2f p2 = vkp2[m.trainIdx].pt;
        Point2f mv = p2 - p1;

        const float dx = mv.x - mean.x;
        const float dy = mv.y - mean.y;
        const float dist = std::sqrt(dx * dx + dy * dy);

        // If motion is close to mean motion, treat as inlier
        if (dist <= thresh || stddev == 0.0f)
        {
            inliersOut.push_back(m);
        }
    }
}


// ==========================================================================
//  CLASS CONSTRUCTOR / DESTRUCTOR / EXECUTE
// ==========================================================================

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
	const std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
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

    std::vector<std::vector<cv::DMatch>> planeMatches(K);

	// Map original matches into planes using labels
	const size_t labelCount = std::min(matchesAll.size(), labels.size());
	for (size_t i = 0; i < labelCount; ++i)
	{
		const int planeId = labels[i];
		if (planeId >= 0 && planeId < K)
		{
			planeMatches[planeId].push_back(matchesAll[i]);
		}
	}

	std::vector<cv::DMatch> aggregatedInliers;
	for (int plane = 0; plane < K; ++plane)
	{
		if (planeMatches[plane].empty())
		{
			continue;
		}

		std::vector<cv::DMatch> planeInliers;
		filterPlaneMatches(planeMatches[plane], vkp1, vkp2, thresholdFactor, planeInliers);

		aggregatedInliers.insert(
			aggregatedInliers.end(),
			planeInliers.begin(),
			planeInliers.end());
	}

    // ---------------------------------------------------------------------
	// Step 3: Aggregate inlier matches from all planes and return
	// ---------------------------------------------------------------------
    if (!aggregatedInliers.empty())
	{
		vDMatches = aggregatedInliers;
	}
	else
	{
		// If nothing survived filtering, fall back to original matches
		vDMatches = matchesAll;
	}
}