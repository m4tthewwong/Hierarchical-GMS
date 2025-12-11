// HGMSMatcher.hpp
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
#ifndef OPENCV_XFEATURES2D_HGMS_HPP
#define OPENCV_XFEATURES2D_HGMS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <utility> // For std::pair

// Include core OpenCV headers that GMS/HGMS relies on
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

namespace cv
{
    namespace xfeatures2d
    {

        /**
         * @brief Hierarchical GMS (HGMS) Matcher class.
         * This class implements the hierarchical filtering logic (e.g., 8x8 -> 16x16 -> 20x20)
         * derived from the original GMSMatcher.
         * * NOTE: The full definition (private members and methods) remains in hgms.cpp.
         */
        class HGMSMatcher
        {
        public:
            /**
             * @brief Constructor for HGMSMatcher.
             * @param vkp1 Keypoints from the first image.
             * @param size1 Size of the first image.
             * @param vkp2 Keypoints from the second image.
             * @param size2 Size of the second image.
             * @param vDMatches Nearest neighbor matches between the two sets of keypoints.
             * @param thresholdFactor Threshold factor used in the GMS score calculation.
             */
             // HGMSMatcher Constructor- Initializing with hierarchical grid sizes
             // precondition: vkp1, vkp2, vDMatches are valid and correspond to each other
             // postcondition: internal structures initialized for hierarchical matching
            HGMSMatcher(const std::vector<KeyPoint>& vkp1, const Size& size1, const std::vector<KeyPoint>& vkp2, const Size& size2,
                const std::vector<DMatch>& vDMatches, const double thresholdFactor);

            // HGMSMatcher Deconstructor - Frees resources when class is deallocated
            // precondition: initialized HGMSMatcher class
            // postcondition: internal structures are freed
            ~HGMSMatcher();

            /**
             * @brief Gets the inlier mask by running the hierarchical GMS filtering process.
             * @param vbInliers Output boolean vector where true indicates an inlier match.
             * @param withRotation Enables checking all 8 rotation patterns.
             * @param withScale Enables checking all 5 scale ratios.
             * @return The total number of inliers found.
             */
             // getInlierMask: Main hierarchical filtering process
             // precondition: internal structures initialized via constructor
             // postcondition: vbInliers filled with final inlier mask, returns total inlier count
            int getInlierMask(std::vector<bool>& vbInliers, const bool withRotation = false, const bool withScale = false);

        private:
            // Private members and implementation details are now defined in the .cpp file.

            std::vector<Point2f> mvP1, mvP2;
            std::vector<std::pair<int, int> > mvMatches;
            size_t mNumberMatches;

            std::vector<Size> mGridSizes;
            Size mCurrentGridSizeLeft, mCurrentGridSizeRight;
            int mCurrentGridNumberLeft;
            int mCurrentGridNumberRight;
            double mThresholdFactor;

            Mat mMotionStatistics;
            std::vector<int> mNumberPointsInPerCellLeft;
            std::vector<int> mCellPairs;
            std::vector<std::pair<int, int> > mvMatchPairs;
            std::vector<bool> mvbInlierMask;
            Mat mGridNeighborLeft;
            Mat mGridNeighborRight;

            // Core methods
            // runSingleStage: Core GMS logic for a single grid size stage
            // precondition: mCurrentGridSizeLeft/Right set, mvbInlierMask holds input mask for this stage
            // postcondition: mvbInlierMask updated with filtered result for this stage, returns inlier count
            int runSingleStage(const int rotationType);

            // --- Utility Methods (Adapted for Hierarchical Grid Sizes) ---
            // Note: Essentially the same as in GMS, but adapted to use mCurrentGridSizeLeft/Right

            // assignMatchPairs: Assign matches to grid cell pairs based on current grid size
            // precondition: mCurrentGridSizeLeft/Right set, mvbInlierMask holds input mask for this stage
            // postcondition: mMotionStatistics and mNumberPointsInPerCellLeft populated for current matches
            void assignMatchPairs(const int GridType);

            // convertMatches: Convert DMatch vector to pair<int,int> vector
            // precondition: vDMatches valid
            // postcondition: vMatches populated with (queryIdx, trainIdx) pairs
            void convertMatches(const std::vector<DMatch>& vDMatches, std::vector<std::pair<int, int> >& vMatches);

            // getGridIndexLeft: Get grid index for a point in the left image based on current grid size and shift type
            // precondition: mCurrentGridSizeLeft set
            // postcondition: returns grid index or -1 if out of bounds
            int getGridIndexLeft(const Point2f& pt, const int type);

            // getGridIndexRight: Get grid index for a point in the right image based on current grid size
            // precondition: mCurrentGridSizeRight set
            // postcondition: returns grid index or -1 if out of bounds
            int getGridIndexRight(const Point2f& pt);

            // getNB9: Get 3x3 neighbor indices for a grid cell index
            // precondition: idx valid for gridSize
            // postcondition: returns vector of 9 neighbor indices (-1 if out of bounds)
            std::vector<int> getNB9(const int idx, const Size& GridSize);

            // initalizeNeighbors: Initialize neighbor matrix for all grid cells
            // precondition: neighbor Mat allocated with correct size, gridSize valid
            // postcondition: neighbor Mat populated with neighbor indices
            void initalizeNeighbors(Mat& neighbor, const Size& GridSize);

            // normalizePoints: Normalize keypoint coordinates to [0,1] range based on image size
            // precondition: kp valid, size valid
            // postcondition: npts populated with normalized points
            void normalizePoints(const std::vector<KeyPoint>& kp, const Size& size, std::vector<Point2f>& npts);

            // setScale: Set the right grid size based on the current scale ratio
            // precondition: mCurrentGridSizeLeft set, scale in [0,4]
            // postcondition: mCurrentGridSizeRight and mCurrentGridNumberRight set, mGridNeighborRight initialized
            void setScale(const int scale);

            // verifyCellPairs: Verify cell pairs based on motion statistics and rotation pattern
            // precondition: mMotionStatistics populated, mCurrentGridNumberLeft/Right set
            // postcondition: mCellPairs updated with verified pairs (-1: no match, -2: invalid match)
            void verifyCellPairs(const int rotationType);
        };


        /**
         * @brief Public function to perform Hierarchical GMS filtering on a set of matches.
         * @param size1 Size of the first image.
         * @param size2 Size of the second image.
         * @param keypoints1 Keypoints from the first image.
         * @param keypoints2 Keypoints from the second image.
         * @param matches1to2 Input matches (usually nearest-neighbor) to be filtered.
         * @param matchesHGMS Output vector containing only the GMS-verified inlier matches.
         * @param withRotation Enables rotation checks (8 patterns).
         * @param withScale Enables scale checks (5 ratios).
         * @param thresholdFactor Multiplier for the score threshold.
         */
         // matchHGMS: Public function to perform HGMS matching
         // precondition: size1, size2, keypoints1, keypoints2, matches1to2 valid
         // postcondition: matchesHGMS populated with filtered matches
        void matchHGMS(const Size& size1, const Size& size2, const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
            const std::vector<DMatch>& matches1to2, std::vector<DMatch>& matchesHGMS, const bool withRotation = false, const bool withScale = false,
            const double thresholdFactor = 6.0);

    } //namespace xfeatures2d
} //namespace cv

#endif // OPENCV_XFEATURES2D_HGMS_HPP