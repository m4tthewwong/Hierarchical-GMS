// hgms.cpp
// This file implements the Hierarchical GMS (HGMS) feature matcher based on the original GMS algorithm.
// It extends the GMS approach by introducing a hierarchical grid structure to improve matching robustness and efficiency.

/*
* The HGMS (Hierarchical Grid-based Motion Statistics) feature matcher implementation shares all the functions
* and structures of the original GMS matcher, with modifications to support hierarchical grid sizes.
*
* The key changes include:
* 1. Hierarchical Grid Sizes: The matcher now defines multiple grid sizes (coarse to fine) to progressively filter matches.
* 2. Progressive Filtering: The inlier mask is updated progressively through each grid stage, allowing for more robust matching.
* 3. Scale Handling: The right image grid size is adjusted based on predefined scale ratios at each hierarchical level.
*
* Functions modified include:
* - HGMSMatcher Constructor: Initializes hierarchical grid sizes and structures.
* - getInlierMask: Implements the hierarchical filtering process across multiple grid sizes.
* - runSingleStage: Core GMS logic adapted for hierarchical grid sizes.
* 
* All other functions from the original GMS matcher are retained with minor adjustments to accommodate the hierarchical approach.
*/

#include "hgms.hpp" // Use the newly created header
#include <iostream>

using namespace std;

namespace cv
{
    namespace xfeatures2d
    {
        // 8 possible rotation and each one is 3 X 3
        const int mRotationPatterns[8][9] = {
            { 1,2,3, 4,5,6, 7,8,9 },
            { 4,1,2, 7,5,3, 8,9,6 },
            { 7,4,1, 8,5,2, 9,6,3 },
            { 8,7,4, 9,5,1, 6,3,2 },
            { 9,8,7, 6,5,4, 3,2,1 },
            { 6,9,8, 3,5,7, 2,1,4 },
            { 3,6,9, 2,5,8, 1,4,7 },
            { 2,3,6, 1,5,9, 4,7,8 }
        };

        // 5 level scales
        const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / std::sqrt(2.0), std::sqrt(2.0), 2.0 };

        // --- HGMSMatcher Implementation ---

		// HGMSMatcher Constructor- Initializing with hierarchical grid sizes
		// precondition: vkp1, vkp2, vDMatches are valid and correspond to each other
		// postcondition: internal structures initialized for hierarchical matching
        HGMSMatcher::HGMSMatcher(const vector<KeyPoint>& vkp1, const Size& size1, const vector<KeyPoint>& vkp2, const Size& size2,
            const vector<DMatch>& vDMatches, const double thresholdFactor) : mThresholdFactor(thresholdFactor)
        {
            // Input initialize
            normalizePoints(vkp1, size1, mvP1);
            normalizePoints(vkp2, size2, mvP2);
            mNumberMatches = vDMatches.size();
            convertMatches(vDMatches, mvMatches);

            // Initialize match pairs storage
            mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

            // Define the hierarchical grid sizes: Coarse -> Medium -> Fine
            mGridSizes.push_back(Size(15, 15));
            mGridSizes.push_back(Size(20, 20));
            mGridSizes.push_back(Size(25, 25)); // Final resolution

            // Initialize the final inlier mask to all true initially
            mvbInlierMask.assign(mNumberMatches, true);
        }

        HGMSMatcher::~HGMSMatcher() {}

		// getInlierMask: Main hierarchical filtering process
		// precondition: internal structures initialized via constructor
		// postcondition: vbInliers filled with final inlier mask, returns total inlier count
        int HGMSMatcher::getInlierMask(vector<bool>& vbInliers, const bool withRotation, const bool withScale)
        {
            // NOTE: mvbInlierMask starts as all 'true' (from constructor) and will hold the running filtered result.
            int final_inlier_count = 0; // Tracks the result of the last (final) stage

            // --- Hierarchical Loop: Iterate from Coarse to Fine Grids (or Fine to Coarse) ---
			int stage_index = 1; // For debug output
            for (const auto& grid_size : mGridSizes)
            {
                mCurrentGridSizeLeft = grid_size;
                mCurrentGridNumberLeft = mCurrentGridSizeLeft.width * mCurrentGridSizeLeft.height;

                mGridNeighborLeft = Mat::zeros(mCurrentGridNumberLeft, 9, CV_32SC1);
                initalizeNeighbors(mGridNeighborLeft, mCurrentGridSizeLeft);

                int current_stage_max_inlier = 0;

                // 1. Store the input mask (result of the previous stage)
                // This is the progressive input for ALL rotation/scale checks in this grid stage.
                const vector<bool> stage_input_mask = mvbInlierMask;
                vector<bool> current_best_mask_in_stage = stage_input_mask;

                int num_scales = withScale ? 5 : 1;
                for (int scale = 0; scale < num_scales; scale++)
                {
                    setScale(scale);

                    int num_rotations = withRotation ? 8 : 1;
                    for (int rotationType = 1; rotationType <= num_rotations; rotationType++)
                    {
                        // 2. RESET: Reset the working mask to the consistent input (result of previous grid stage).
                        mvbInlierMask = stage_input_mask;

                        int num_inlier = runSingleStage(rotationType);

                        // 3. TRACK: Track the best result found in this stage (across all rotations/scales).
                        if (num_inlier > current_stage_max_inlier)
                        {
                            current_stage_max_inlier = num_inlier;
                            current_best_mask_in_stage = mvbInlierMask; // Store the resulting mask
                        }
                    }
                }

                // 4. TRANSITION: Set the output mask for the current stage as the input for the next (finer/coarser) stage.
                // This is the core progressive filtering update.
                mvbInlierMask = current_best_mask_in_stage;
                final_inlier_count = current_stage_max_inlier;

                // 5. DEBUG:
                //std::cout << "[HGMS DEBUG] Stage " << stage_index++ << ": Grid "
                //    << mCurrentGridSizeLeft.width << "x" << mCurrentGridSizeLeft.height
                //    << " filtered. Inliers remaining: " << current_stage_max_inlier << std::endl;
            }

            // The result of the final stage is the final output.
            vbInliers = mvbInlierMask;
            return final_inlier_count;
        }

		// runSingleStage: Core GMS logic for a single grid size stage
		// precondition: mCurrentGridSizeLeft/Right set, mvbInlierMask holds input mask for this stage
		// postcondition: mvbInlierMask updated with filtered result for this stage, returns inlier count
        int HGMSMatcher::runSingleStage(const int rotationType)
        {
            // The size of the matrix depends on the currently set grid
            mMotionStatistics = Mat::zeros(mCurrentGridNumberLeft, mCurrentGridNumberRight, CV_32SC1);
            vector<bool> vbTempInliers(mNumberMatches, false); // Tracks if a match passes ANY of the 4 shifts

            for (int gridType = 1; gridType <= 4; gridType++)
            {
                // Reset for this shift
                mMotionStatistics.setTo(0);
                mCellPairs.assign(mCurrentGridNumberLeft, -1);
                mNumberPointsInPerCellLeft.assign(mCurrentGridNumberLeft, 0);

                // Assignment loop (only for current inliers, using mvbInlierMask)
                for (size_t i = 0; i < mNumberMatches; i++)
                {
                    if (!mvbInlierMask[i]) continue; // Only consider matches still marked as inliers

                    Point2f& lp = mvP1[mvMatches[i].first];
                    Point2f& rp = mvP2[mvMatches[i].second];
                    int lgidx = mvMatchPairs[i].first = getGridIndexLeft(lp, gridType);
                    int rgidx = mvMatchPairs[i].second = getGridIndexRight(rp);

                    if (lgidx < 0 || rgidx < 0) continue;

                    mMotionStatistics.at<int>(lgidx, rgidx)++;
                    mNumberPointsInPerCellLeft[lgidx]++;
                }

                verifyCellPairs(rotationType);

                // Mark inliers for THIS SHIFT (OR logic with previous shifts in this stage)
                for (size_t i = 0; i < mNumberMatches; i++)
                {
                    if (!mvbInlierMask[i]) continue;

                    if (mvMatchPairs[i].first >= 0 && mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
                        vbTempInliers[i] = true; // Passed for at least one shift in this stage
                }
            }

            // Update the master inlier mask (AND logic with the current stage result)
            int num_inliers = 0;
            for (size_t i = 0; i < mNumberMatches; i++)
            {
                // A match must have passed the previous stages (mvbInlierMask[i]) AND passed the current stage (vbTempInliers[i])
                if (mvbInlierMask[i] && vbTempInliers[i])
                {
                    mvbInlierMask[i] = true;
                    num_inliers++;
                }
                else
                {
                    mvbInlierMask[i] = false;
                }
            }
            return num_inliers;
        }

        // --- Utility Methods (Adapted for Hierarchical Grid Sizes) ---
		// Note: Essentially the same as in GMS, but adapted to use mCurrentGridSizeLeft/Right

		// assignMatchPairs: Assign matches to grid cell pairs based on current grid size
		// precondition: mCurrentGridSizeLeft/Right set, mvbInlierMask holds input mask for this stage
		// postcondition: mMotionStatistics and mNumberPointsInPerCellLeft populated for current matches
        void HGMSMatcher::assignMatchPairs(const int gridType)
        {
            // Note: This function's logic is largely incorporated into runSingleStage in HGMS
            // to efficiently handle the filtering of points using mvbInlierMask.
            for (size_t i = 0; i < mNumberMatches; i++)
            {
                Point2f& lp = mvP1[mvMatches[i].first];
                Point2f& rp = mvP2[mvMatches[i].second];

                int lgidx = mvMatchPairs[i].first = getGridIndexLeft(lp, gridType);
                int rgidx = mvMatchPairs[i].second = getGridIndexRight(rp);

                if (lgidx < 0 || rgidx < 0) continue;

                mMotionStatistics.at<int>(lgidx, rgidx)++;
                mNumberPointsInPerCellLeft[lgidx]++;
            }
        }

		// convertMatches: Convert DMatch vector to pair<int,int> vector
		// precondition: vDMatches valid
		// postcondition: vMatches populated with (queryIdx, trainIdx) pairs
        void HGMSMatcher::convertMatches(const vector<DMatch>& vDMatches, vector<pair<int, int> >& vMatches)
        {
            vMatches.resize(mNumberMatches);
            for (size_t i = 0; i < mNumberMatches; i++)
                vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
        }

		// getGridIndexLeft: Get grid index for a point in the left image based on current grid size and shift type
		// precondition: mCurrentGridSizeLeft set
		// postcondition: returns grid index or -1 if out of bounds
        int HGMSMatcher::getGridIndexLeft(const Point2f& pt, const int type)
        {
            int x = 0, y = 0;

            if (type == 1) {
                x = cvFloor(pt.x * mCurrentGridSizeLeft.width);
                y = cvFloor(pt.y * mCurrentGridSizeLeft.height);
            }
            else if (type == 2) {
                x = cvFloor(pt.x * mCurrentGridSizeLeft.width + 0.5);
                y = cvFloor(pt.y * mCurrentGridSizeLeft.height);
            }
            else if (type == 3) {
                x = cvFloor(pt.x * mCurrentGridSizeLeft.width);
                y = cvFloor(pt.y * mCurrentGridSizeLeft.height + 0.5);
            }
            else if (type == 4) {
                x = cvFloor(pt.x * mCurrentGridSizeLeft.width + 0.5);
                y = cvFloor(pt.y * mCurrentGridSizeLeft.height + 0.5);
            }

            if (x >= mCurrentGridSizeLeft.width || y >= mCurrentGridSizeLeft.height)
                return -1;

            return x + y * mCurrentGridSizeLeft.width;
        }

		// getGridIndexRight: Get grid index for a point in the right image based on current grid size
		// precondition: mCurrentGridSizeRight set
		// postcondition: returns grid index or -1 if out of bounds
        int HGMSMatcher::getGridIndexRight(const Point2f& pt)
        {
            int x = cvFloor(pt.x * mCurrentGridSizeRight.width);
            int y = cvFloor(pt.y * mCurrentGridSizeRight.height);

            if (x < 0 || x >= mCurrentGridSizeRight.width || y < 0 || y >= mCurrentGridSizeRight.height)
                return -1; // Added bounds check for safety

            return x + y * mCurrentGridSizeRight.width;
        }

		// getNB9: Get 3x3 neighbor indices for a grid cell index
		// precondition: idx valid for gridSize
		// postcondition: returns vector of 9 neighbor indices (-1 if out of bounds)
        vector<int> HGMSMatcher::getNB9(const int idx, const Size& gridSize)
        {
            vector<int> NB9(9, -1);

            int idx_x = idx % gridSize.width;
            int idx_y = idx / gridSize.width;

            for (int yi = -1; yi <= 1; yi++)
            {
                for (int xi = -1; xi <= 1; xi++)
                {
                    int idx_xx = idx_x + xi;
                    int idx_yy = idx_y + yi;

                    if (idx_xx < 0 || idx_xx >= gridSize.width || idx_yy < 0 || idx_yy >= gridSize.height)
                        continue;

                    // The index calculation in the original GMS code: NB9[xi + 4 + yi * 3] 
                    // This maps the 3x3 neighbor to the 1D index: 
                    // (-1,-1)=0, (0,-1)=1, (1,-1)=2, (-1,0)=3, (0,0)=4, (1,0)=5, (-1,1)=6, (0,1)=7, (1,1)=8
                    NB9[xi + 1 + (yi + 1) * 3] = idx_xx + idx_yy * gridSize.width;
                }
            }
            return NB9;
        }

		// initalizeNeighbors: Initialize neighbor matrix for all grid cells
		// precondition: neighbor Mat allocated with correct size, gridSize valid
		// postcondition: neighbor Mat populated with neighbor indices
        void HGMSMatcher::initalizeNeighbors(Mat& neighbor, const Size& gridSize)
        {
            for (int i = 0; i < neighbor.rows; i++)
            {
                vector<int> NB9 = getNB9(i, gridSize);
                int* data = neighbor.ptr<int>(i);
                memcpy(data, &NB9[0], sizeof(int) * 9);
            }
        }

		// normalizePoints: Normalize keypoint coordinates to [0,1] range based on image size
		// precondition: kp valid, size valid
		// postcondition: npts populated with normalized points
        void HGMSMatcher::normalizePoints(const vector<KeyPoint>& kp, const Size& size, vector<Point2f>& npts)
        {
            const size_t numP = kp.size();
            const int width = size.width;
            const int height = size.height;
            npts.resize(numP);

            for (size_t i = 0; i < numP; i++)
            {
                npts[i].x = kp[i].pt.x / width;
                npts[i].y = kp[i].pt.y / height;
            }
        }

		// setScale: Set the right grid size based on the current scale ratio
		// precondition: mCurrentGridSizeLeft set, scale in [0,4]
		// postcondition: mCurrentGridSizeRight and mCurrentGridNumberRight set, mGridNeighborRight initialized
        void HGMSMatcher::setScale(const int scale)
        {
            // mCurrentGridSizeLeft is set in the hierarchical loop
            mCurrentGridSizeRight.width = cvRound(mCurrentGridSizeLeft.width * mScaleRatios[scale]);
            mCurrentGridSizeRight.height = cvRound(mCurrentGridSizeLeft.height * mScaleRatios[scale]);
            mCurrentGridNumberRight = mCurrentGridSizeRight.width * mCurrentGridSizeRight.height;

            // Initialize the neighbor of right grid
            mGridNeighborRight = Mat::zeros(mCurrentGridNumberRight, 9, CV_32SC1);
            initalizeNeighbors(mGridNeighborRight, mCurrentGridSizeRight);
        }

		// verifyCellPairs: Verify cell pairs based on motion statistics and rotation pattern
		// precondition: mMotionStatistics populated, mCurrentGridNumberLeft/Right set
		// postcondition: mCellPairs updated with verified pairs (-1: no match, -2: invalid match)
        void HGMSMatcher::verifyCellPairs(const int rotationType)
        {
            const int* CurrentRP = mRotationPatterns[rotationType - 1];

            for (int i = 0; i < mCurrentGridNumberLeft; i++)
            {
                if (sum(mMotionStatistics.row(i))[0] == 0)
                {
                    mCellPairs[i] = -1;
                    continue;
                }

                int max_number = 0;
                for (int j = 0; j < mCurrentGridNumberRight; j++)
                {
                    int* value = mMotionStatistics.ptr<int>(i);
                    if (value[j] > max_number)
                    {
                        mCellPairs[i] = j;
                        max_number = value[j];
                    }
                }

                int idx_grid_rt = mCellPairs[i];

                const int* NB9_lt = mGridNeighborLeft.ptr<int>(i);
                const int* NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

                int score = 0;
                double thresh = 0;
                int numpair = 0;

                for (size_t j = 0; j < 9; j++)
                {
                    int ll = NB9_lt[j];
                    int rr = NB9_rt[CurrentRP[j] - 1];
                    if (ll == -1 || rr == -1)
                        continue;

                    score += mMotionStatistics.at<int>(ll, rr);
                    thresh += mNumberPointsInPerCellLeft[ll];
                    numpair++;
                }

                if (numpair == 0) { // Avoid division by zero
                    mCellPairs[i] = -2;
                    continue;
                }

                thresh = mThresholdFactor * std::sqrt(thresh / numpair);

                if (score < thresh)
                    mCellPairs[i] = -2;
            }
        }

        // --- Public Function Implementation ---

		// matchHGMS: Public function to perform HGMS matching
		// precondition: size1, size2, keypoints1, keypoints2, matches1to2 valid
		// postcondition: matchesHGMS populated with filtered matches
        void matchHGMS(const Size& size1, const Size& size2, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
            const vector<DMatch>& matches1to2, vector<DMatch>& matchesHGMS, const bool withRotation, const bool withScale,
            const double thresholdFactor)
        {
            HGMSMatcher hgms(keypoints1, size1, keypoints2, size2, matches1to2, thresholdFactor);
            vector<bool> inlierMask;
            hgms.getInlierMask(inlierMask, withRotation, withScale);

            matchesHGMS.clear();
            for (size_t i = 0; i < inlierMask.size(); i++) {
                if (inlierMask[i])
                    matchesHGMS.push_back(matches1to2[i]);
            }
        }

    } //namespace xfeatures2d
} //namespace cv