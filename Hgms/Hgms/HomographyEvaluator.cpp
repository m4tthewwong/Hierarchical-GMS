//---------------------------------------------------------------------------  
// HomographyEvaluator.cpp
// The HomographyEvaluator class file is for consuming image datasets and then comparing
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
#include<fstream>
#include "HomographyEvaluator.h"

/*----------------------------- default -----------------------------------
* Default constructor for class BenchmarkTests.
* Preconditions: none
* Postconditions: Instance of BenchmarkTests is instantiated.
*/
HomographyEvaluator::HomographyEvaluator()
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class BenchmarkTests.
* Preconditions: Instance of ProcessingStage class is instantiated.
* Postconditions: Objects are deallocated.
*/
HomographyEvaluator::~HomographyEvaluator()
{
}

/*----------------------------- computeBenchmarks ------------------------
* Public method to return benchmark calculated metrics related to how well matches
* are filtered.
* Preconditions: Valid mat file representing loaded homography file, keypoints, and matches
* Postconditions: This method will return the BenchmarkMetrics data struct
*/
HomographyEvaluator::HomographyMetrics HomographyEvaluator::computeBenchmarks(
	const std::vector<KeyPoint>& kp1,
	const std::vector<KeyPoint>& kp2,
	const std::vector<DMatch>& matches,
	const Mat& hMat,
	const float threshold)
{
	// convert keypoints to points for applying homography
	std::vector<Point2f> sourcePts, destPts;
	KeyPoint::convert(kp1, sourcePts);
	KeyPoint::convert(kp2, destPts);

	// apply transform with homography
	std::vector<Point2f> hPoints = applyTransform(sourcePts, hMat);

	// compute true/false positives
	HomographyMetrics metrics;
	computeTrueFalsePositives(hPoints, destPts, matches, threshold, metrics);

	// update and return benchmarks
	computeStatistics(metrics);

	return metrics;
}

/*----------------------------- loadHomographyFile ------------------------
* Method to load homography file to calculate transforms
* Preconditions: Valid path to homography file and 3x3 CV_64F Mat object
* Postconditions: This method will return a Mat with the loaded 3x3 homography and
*                 true is returned. If the file is invalid then no Mat is
*                 returned and false is returned
*/
Mat HomographyEvaluator::loadHomographyFile(const std::string filepath)
{
	bool isValidFile = false;
	int rowColWidth = 3;

	// load homography 3x3 from file into Mat to use for transformation
	std::ifstream hFile(filepath);

	// check if file loaded successfully and if not return false;
	if (!hFile.is_open())
	{
		std::cerr << "Unable to load homography file: " << filepath << std::endl;
		return Mat();
	}

	// initialized Mat with 3 x 3 and 64 bit
	Mat hMat(rowColWidth, rowColWidth, CV_64F);

	// try parsing row/col contents into Mat
	for (int row = 0; row < rowColWidth; row++)
	{
		for (int col = 0; col < rowColWidth; col++)
		{
			if (!(hFile >> hMat.at<double>(row, col)))
			{
				// error
				std::cerr << "Error parsing homography file: " << filepath << std::endl;
				return Mat();
			}
		}
	}

	return hMat;
}

/*----------------------------- applyTransform ------------------------
* Private helper method to apply warp to keypoints based on loaded homography file
* Preconditions: Valid keypoints and Mat from homography file
* Postconditions: This method will return a point corresponding to the warped point
*/
const std::vector<Point2f> HomographyEvaluator::applyTransform(const std::vector<Point2f>& sourcePoints, const Mat& hMat)
{
	// apply homography mat to source point using perspectiveTransform which
	// will apply and normalize from 3d to 2d.
	std::vector<Point2f> transformedPoints;
	perspectiveTransform(sourcePoints, transformedPoints, hMat);

	return transformedPoints;
}

/*----------------------------- computeStatistics ------------------------
* Private helper method to compute the statistics for precision, recall, fscore
* Preconditions: Benchmark Metrics struct containing true positives and false positives
* Postconditions: This method will return the updated Benchmark Metrics struct
*                 with calculated precision, recall, and fscore
*/
void HomographyEvaluator::computeStatistics(HomographyMetrics& metrics)
{
	// using true positives and false positives, calculate precision, recall, fscore
	// Precision = TP / (TP + FP)
	float precision = metrics.truePositives / 
		(static_cast<float>(metrics.truePositives + metrics.falsePositives));

	// Recall = TP / (TP + FN)
	float recall = metrics.truePositives / 
		(static_cast<float>(metrics.truePositives + metrics.falseNegatives));

	// Fscore = 2 *  (Precision * Recall) / (Precision + Recall)
	float fscore = 2 * (precision * recall) / (precision + recall);

	// Update benchmarks
	metrics.precision = (precision >= 0) ? precision : 0;
	metrics.recall = (recall >= 0) ? recall : 0;
	metrics.fscore = (fscore >= 0) ? fscore : 0;
}

/*----------------------------- computeTrueFalsePositives ------------------------
* Private helper method to compute the true and false positives for matches
* Preconditions: Benchmark Metrics struct
* Postconditions: This method will return the updated Benchmark Metrics struct
*                 with the count of true and false positives
*/
void HomographyEvaluator::computeTrueFalsePositives(const std::vector<Point2f>& hPoints, const std::vector<Point2f>& actualPoints, 
	const std::vector<DMatch>& matches, const float threshold, HomographyMetrics& metrics)
{
	// initialize true and false positive counters
	int truePositives = 0, falsePositives = 0;

	// iterate matches and determine if distance between the predicted destination
	// and tranformed destination are within threshold. If within threshold then 
	// consider the match a true positive else the match is a false positive
	for (auto& match : matches)
	{
		Point2f predPoint = hPoints[match.queryIdx];
		Point2f actualPoint = actualPoints[match.trainIdx];

		// calculate if distance between predicted ground truth and actual point
		double distance = norm(predPoint - actualPoint);

		// if distance less than threshold consider the match good
		if (distance <= threshold)
		{
			truePositives++;
		}
		else
		{
			falsePositives++;
		}
	}

	// return benchMark struct with true and false positives
	metrics.truePositives = truePositives;
	metrics.falsePositives = falsePositives;
	metrics.falseNegatives = hPoints.size() - truePositives;
}
