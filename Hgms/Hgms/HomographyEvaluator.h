//---------------------------------------------------------------------------  
// HomographyEvaluator.h
// The HomographyEvaluator header file is for consuming image datasets and then comparing
// performance of HGMS against GMS. This class supports determining
// the following metrics: precision, recall, f-measure for the VGG image dataset.
// The VGG image dataset was selected as this dataset provide images with different
// affine transformations, includes a homography file for each image dataset that
// can be used as the ground truth.
// 
// the following operations:
//	1. 
// Authors:  Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- VGG affine transformation dataset and related homography files
//     https://www.robots.ox.ac.uk/~vgg/research/affine/
// 
// Outputs:
// -- computation of precision, recall, f-measure for each image pair in comma
//    delimited format. The precision, recall, f-measure will be computed for 
//    GMS and HGMS.
// 
// Description:
//    This class provides methods to read, calculate, and determine the recall,
//    f-measure, for VGG image datasets.
//
// Assumptions:
//   -- VGG affine covariant regions dataset is available in location of exe
// 
#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>

using namespace cv;

class HomographyEvaluator {
public:
	// struct to return calculated results
	struct HomographyMetrics {
		size_t truePositives;
		size_t falsePositives;
		size_t falseNegatives;
		double fscore;
		double recall;
		double precision;
	};

	/*----------------------------- default -----------------------------------
	* Default constructor for class HomographyEvaluator.
	* Preconditions: none
	* Postconditions: Instance of HomographyEvaluator is instantiated.
	*/
	HomographyEvaluator();

	/*----------------------------- destructor -------------------------------
	* Destructor for class HomographyEvaluator.
	* Preconditions: Instance of HomographyEvaluator class is instantiated.
	* Postconditions: Objects are deallocated.
	*/
	~HomographyEvaluator();

	/*----------------------------- computeBenchmarks ------------------------
	* Public method to return benchmark calculated metrics related to how well matches
	* are filtered.
	* Preconditions: Valid mat file representing loaded homography file, keypoints, 
	*  and matches
	* Postconditions: This method will return the BenchmarkMetrics data struct
	*/
	HomographyMetrics computeBenchmarks(const std::vector<KeyPoint>& kp1,
									   const std::vector<KeyPoint>& kp2,
									   const std::vector<DMatch>& matches,
									   const Mat& hMat,
									   const float threshold = 3.0f);

	/*----------------------------- loadHomographyFile ------------------------
	* Method to load homography file to calculate transforms
	* Preconditions: Valid path to homography file
	* Postconditions: This method will return a Mat with the loaded 3x3 homography.
	*                 If an invalid file is provided then an empty mat is returned.
	*/
	Mat loadHomographyFile(const std::string filepath);

private:
	/*----------------------------- applyTransform ------------------------
	* Private helper method to apply warp to keypoints based on loaded homography file
	* Preconditions: Valid keypoints and Mat from homography file
	* Postconditions: This method will return a point corresponding to the warped point
	*/
	const std::vector<Point2f> applyTransform(const std::vector<Point2f>& sourcePoints, const Mat& hMat);

	/*----------------------------- computeStatistics ------------------------
	* Private helper method to compute the statistics for precision, recall, fscore
	* Preconditions: Benchmark Metrics struct containing true positives and false positives
	* Postconditions: This method will return the updated Benchmark Metrics struct
	*                 with calculated precision, recall, and fscore
	*/
	void computeStatistics(HomographyMetrics& metrics);

	/*----------------------------- computeTrueFalsePositives ------------------------
	* Private helper method to compute the true and false positives for matches
	* Preconditions: Benchmark Metrics struct
	* Postconditions: This method will return the updated Benchmark Metrics struct
	*                 with the count of true and false positives
	*/
	void computeTrueFalsePositives(const std::vector<Point2f>& hPoints, const std::vector<Point2f>& actualPoints,
		const std::vector<DMatch>& matches, const float threshold, HomographyMetrics& metrics);

};