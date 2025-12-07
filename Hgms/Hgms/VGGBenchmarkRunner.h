//---------------------------------------------------------------------------
// VGGBenchmarkRunner.h
// Header code for runner class that compares performance of the HGMS algorithm 
// against GMS algorithm for the VGG affine dataset.
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

#include <string>
#include <vector>
#include "HomographyEvaluator.h"
#include "HGMSStage.h"
#include "LATStage.h"
#include "MPStage.h"

class VGGBenchmarkRunner {
public:
	/*----------------------------- default -----------------------------------
	* Default constructor for class VGGBenchmarkRunner.
	* Preconditions: none
	* Postconditions: Instance of VGGBenchmarkRunner is instantiated.
	*/
	VGGBenchmarkRunner();

	/*----------------------------- destructor -------------------------------
	* Destructor for class HomographyEvaluator.
	* Preconditions: Instance of HomographyEvaluator class is instantiated.
	* Postconditions: Objects are deallocated.
	*/
	~VGGBenchmarkRunner();
	
	/*----------------------------- run -------------------------------
	* Run method to execute tests of VGG dataset and generate comparison metrics
	* Preconditions: Instance of HomographyEvaluator class is instantiated and valid
	*                path to VGG dataset files. Also specified are the number of max features
	*                to be used in the comparison and the threshold for evaluating matches
	*                against groundtruth.
	* Postconditions: Tests are completed and metrics generated.
	*/
	void run(const std::string imagesetName, const std::string vggFolderpath, const int maxFeatures,
		const float evaluateThreshold);

	/*----------------------------- printResults -------------------------------
	* Method to print benchmark metric results from tests
	* Preconditions: Instance of HomographyEvaluator class is instantiated and run method
	*                has been called successfully.
	* Postconditions: Test results are printed to screen.
	*/
	void printResults() const;
	
	/*----------------------------- printCsvSummaryResults -------------------------------
	* Method to print average benchmark metric results from tests for each set of VGG images
	* that can then be plotted.
	* Preconditions: Instance of HomographyEvaluator class is instantiated and run method
	*                has been called successfully.
	* Postconditions: Test results in comma delimited format are printed to screen.
	*/
	void printCsvSummaryResults() const;

private:
	// constants for the HGMS and GMS threshold
	const float HGMS_THRESHOLD = 6.0f;
	const float GMS_THRESHOLD = 6.0f;

	// struct to hold relationship between VGG image sets and related homography file
	struct ImageSet {
		std::string image1;
		std::string image2;
		std::string homography;
	};

	// struct to hold results for specific algorithm;
	struct ExecutionResult {
		std::string imagesetName;
		std::string algoName;
		ImageSet imageset;
		HomographyEvaluator::HomographyMetrics metrics;
	};

	// vector store VGG dataset to process. The naming, number of images, homography
	// of the VGG datasets are identical between image sets.
	std::vector<ImageSet> vggDataSets = {
		{"img1.ppm", "img2.ppm", "H1to2p"},
		{"img1.ppm", "img3.ppm", "H1to3p"},
		{"img1.ppm", "img4.ppm", "H1to4p"},
		{"img1.ppm", "img5.ppm", "H1to5p"},
		{"img1.ppm", "img6.ppm", "H1to6p"}
	};
	
	// vector to store results from each Algorithm (none, GMS, HGMS) for an image batch 
	std::vector<ExecutionResult> results;

	/*----------------------------- isValidImage --------------------------
	 * Helper method to validate that the image provided is a valid image.
	 * Preconditions: Valid path to image1 and image2 files from imageset and number of features
	 *                to calculate.
	 * Postconditions: Calculated keypoints for each image and matches using
	 *  bruteforce matcher. At false will be returned if there are errors in images.
	 */
	const bool isValidImage(const Mat& image);

	/*----------------------------- calculateImagesetMatches -------------------------------
	* Method to calculate and return keypoints and matches for imageset
	* Preconditions: Valid path to image1 and image2 files from imageset and number of features
	*                to calculate.
	* Postconditions: Calculated keypoints for each image and matches using
	*  bruteforce matcher.
	*/
	bool calculateImagesetMatches(const std::string image1, Size& size1,
		const std::string image2, Size& size2, const int maxFeatures,
		std::vector<KeyPoint>& kp1, std::vector<KeyPoint>& kp2, std::vector<DMatch>& matches);
	
};