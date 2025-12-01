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
            HGMSMatcher(const std::vector<KeyPoint>& vkp1, const Size& size1, const std::vector<KeyPoint>& vkp2, const Size& size2,
                const std::vector<DMatch>& vDMatches, const double thresholdFactor);

            ~HGMSMatcher();

            /**
             * @brief Gets the inlier mask by running the hierarchical GMS filtering process.
             * @param vbInliers Output boolean vector where true indicates an inlier match.
             * @param withRotation Enables checking all 8 rotation patterns.
             * @param withScale Enables checking all 5 scale ratios.
             * @return The total number of inliers found.
             */
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
            int runSingleStage(const int rotationType);
            void assignMatchPairs(const int GridType);
            void convertMatches(const std::vector<DMatch>& vDMatches, std::vector<std::pair<int, int> >& vMatches);
            int getGridIndexLeft(const Point2f& pt, const int type);
            int getGridIndexRight(const Point2f& pt);
            std::vector<int> getNB9(const int idx, const Size& GridSize);
            void initalizeNeighbors(Mat& neighbor, const Size& GridSize);
            void normalizePoints(const std::vector<KeyPoint>& kp, const Size& size, std::vector<Point2f>& npts);
            void setScale(const int scale);
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
        void matchHGMS(const Size& size1, const Size& size2, const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
            const std::vector<DMatch>& matches1to2, std::vector<DMatch>& matchesHGMS, const bool withRotation = false, const bool withScale = false,
            const double thresholdFactor = 6.0);

    } //namespace xfeatures2d
} //namespace cv

#endif // OPENCV_XFEATURES2D_HGMS_HPP