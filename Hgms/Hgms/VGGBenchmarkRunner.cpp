//---------------------------------------------------------------------------
// VGGBenchmarkRunner.cpp
// Class implementation for runner class that compares performance of the HGMS algorithm 
// against GMS algorithm for the VGG affine covariant dataset. 
// https://www.robots.ox.ac.uk/~vgg/research/affine/
// Expectation is that the image files and homography files follow the same naming
// standard as what is in the VGG affine covariant dataset. For each specified
// folder, the expectation is that there are 6 images and 6 homography files.
// The images need to be named img1 - 6 and are in ppm format. The homography
// files are named like the following H1to2p without a file extension and indicate
// the homography transform between 2 sets of images.
// 
// Performs the following operations:
//	1. 
// Authors:  Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- Valid path to VGG affine transform homography and image input files
// 
// Outputs:
// -- Utilizes mock data for generating images, keypoints, matches and outputs
//    metrics to the screen regarding matches and processing time.
// 
// Description:
//    The purpose of this driver file is to assist with testing of individual
//    HGMS classes and functions.
//
// Assumptions:
//   -- None
#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "VGGBenchmarkRunner.h"
#include "HGMSPipeline.h"

/*----------------------------- default -----------------------------------
* Default constructor for class VGGBenchmarkRunner.
* Preconditions: none
* Postconditions: Instance of VGGBenchmarkRunner is instantiated.
*/
VGGBenchmarkRunner::VGGBenchmarkRunner()
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class HomographyEvaluator.
* Preconditions: Instance of HomographyEvaluator class is instantiated.
* Postconditions: Objects are deallocated.
*/
VGGBenchmarkRunner::~VGGBenchmarkRunner()
{
}

/*----------------------------- isValidImage --------------------------
 * Helper method to validate that the image provided is a valid image.
 * Preconditions: Image reference is provided as input
 * Postconditions: A true or false is returned with true = valid image and false
 *                 indicating invalid image.
 */
const bool VGGBenchmarkRunner::isValidImage(const Mat& image)
{
	// check that the image is not empty
	return (!image.empty());
}

/*----------------------------- calculateImagesetMatches -------------------------------
* Method to calculate and return keypoints and matches for imageset
* Preconditions: Valid path to image1 and image2 files from imageset and number of features
*                to calculate.
* Postconditions: Calculated keypoints for each image and matches using
*  bruteforce matcher. At false will be returned if there are errors in images.
*/
bool VGGBenchmarkRunner::calculateImagesetMatches(const std::string image1, Size& size1, 
	const std::string image2, Size& size2, const int maxFeatures, 
	std::vector<KeyPoint>& kp1, std::vector<KeyPoint>& kp2, std::vector<DMatch>& matches)
{
	bool isValidImageset = false;

	// read images
	const Mat imageMat1 = imread(image1, IMREAD_GRAYSCALE);
	const Mat imageMat2 = imread(image2, IMREAD_GRAYSCALE);
	size1 = imageMat1.size();
	size2 = imageMat2.size();

	// check for valid images
	if (!isValidImage(imageMat1) || !isValidImage(imageMat2))
	{
		std::cerr << "Unable to load images. Please check that VGG images exist in the" 
			<< "specified filepath\r\n";
		return isValidImageset;
	}

	isValidImageset = true;

	// create keypoints and descriptors
	std::vector<KeyPoint> k1kp, k2kp;
	Mat k1d, k2d;

	// create features
	Ptr<Feature2D> k1detector = ORB::create(maxFeatures);
	k1detector->detectAndCompute(imageMat1, noArray(), kp1, k1d);
	Ptr<Feature2D> k2detector = ORB::create(maxFeatures);
	k2detector->detectAndCompute(imageMat2, noArray(), kp2, k2d);

	// compute matches
	// Create brute-force matcher to compute matches between descriptors
	Ptr<BFMatcher> matcher = BFMatcher::create(cv::NORM_HAMMING, false);
	std::vector<DMatch> matchesAll;
	matcher->match(k1d, k2d, matches);

	return isValidImageset;
}

/*----------------------------- run -------------------------------
* Run method to execute tests of VGG dataset and generate comparison metrics
* Preconditions: Instance of HomographyEvaluator class is instantiated and valid
*                path to VGG dataset files. Also specified are the number of max features
*                to be used in the comparison and the threshold for evaluating matches
*                against groundtruth.
* Postconditions: Tests are completed and metrics generated.
*/
void VGGBenchmarkRunner::run(const std::string vggFolderpath, const int maxFeatures, 
	const float evaluateThreshold)
{
	// for each imageset in the specified vggFilepath, run comparison and generate metrics
	for (ImageSet set : vggDataSets)
	{
		// format file paths for image loading/processing
		bool isValidImageset = false;
		std::string image1Filepath = vggFolderpath + "\\" + set.image1;
		std::string image2Filepath = vggFolderpath + "\\" + set.image2;
		std::string homographyFilepath = vggFolderpath + "\\" + set.homography;

		// calculate keypoints and matches between images in image set
		std::vector<KeyPoint> imageKp1, imageKp2;
		std::vector<DMatch> matches;
		Size imageSize1, imageSize2;

		// calculate keypoints and feature matches
		isValidImageset = calculateImagesetMatches(image1Filepath, imageSize1,
			image2Filepath, imageSize2, maxFeatures, imageKp1, imageKp2, matches);
		
		// evalute baseline Homography from orb/bruteforce
		HomographyEvaluator evaluator;
		Mat hMat = evaluator.loadHomographyFile(homographyFilepath);

		// if images are valid and homography is valid
		if (isValidImageset && !hMat.empty())
		{
			// Evaluate baseline with no filtering only ORB/Bruteforce
			HomographyEvaluator::HomographyMetrics baselineMetrics = 
				evaluator.computeBenchmarks(imageKp1, imageKp2, matches, hMat, evaluateThreshold);
			ExecutionResult baselineResult = { "Baseline", set, baselineMetrics };
			results.emplace_back(baselineResult);

			// generate filtered matches from GMS
			// false for rotation and scale and use default thresholding of 6.0
			std::vector<DMatch> gmsFilteredMatches;
			xfeatures2d::matchGMS(imageSize1, imageSize2, imageKp1, imageKp2, matches, gmsFilteredMatches);
			// evalute Homography from orb/bruteforce filtered through GMS
			HomographyEvaluator::HomographyMetrics gmsMetrics =
				evaluator.computeBenchmarks(imageKp1, imageKp2, gmsFilteredMatches, hMat, evaluateThreshold);
			ExecutionResult gmsResult = { "GMS", set, gmsMetrics };
			results.emplace_back(gmsResult);

			// generate filtered matches from HGMS
			std::vector<DMatch> hgmsFilteredMatches;
			HGMSPipeline allStagePipeline(HGMSPipeline::FILTER);
			allStagePipeline.addStage(std::make_shared<HGMSStage>());
			allStagePipeline.addStage(std::make_shared<LATStage>());
			allStagePipeline.addStage(std::make_shared<MPStage>());
			allStagePipeline.match(imageKp1, imageSize1, imageKp2, imageSize2, matches,
				hgmsFilteredMatches, HGMS_THRESHOLD);

			// evalute Homography from orb/bruteforce filtered through HGMS
			HomographyEvaluator::HomographyMetrics hgmsMetrics =
				evaluator.computeBenchmarks(imageKp1, imageKp2, hgmsFilteredMatches, hMat, evaluateThreshold);
			ExecutionResult hgmsResult = { "HGMS", set, hgmsMetrics };
			results.emplace_back(hgmsResult);

		}
	}
}

/*----------------------------- printResults -------------------------------
* Method to print benchmark metric results from tests
* Preconditions: Instance of HomographyEvaluator class is instantiated and run method
*                has been called successfully.
* Postconditions: Test results are printed to screen.
*/
void VGGBenchmarkRunner::printResults() const
{
	for (ExecutionResult result : results)
	{
		std::cout << "----------------------------" << std::endl;
		std::cout << result.algoName << std::endl;
		std::cout << "ImageSet Image1: " << result.imageset.image1 << " Image2: "
			<< result.imageset.image2 << " Homography: " << result.imageset.homography << std::endl;
		std::cout << "Precision: " << result.metrics.precision << std::endl;
		std::cout << "Recall: " << result.metrics.recall << std::endl;
		std::cout << "FScore: " << result.metrics.fscore << std::endl;
	}
}