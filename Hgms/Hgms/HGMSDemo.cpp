//---------------------------------------------------------------------------
// HGMSDemo.cpp
// Driver code for testing the HGMS algorithm and associated class files
// Performs the following operations:
//	1. Find matches by local affine transformation
// Authors:  Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- None
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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "HGMSUnitTests.h"
#include "VGGBenchmarkRunner.h"

// Constants for determining if unit tests and benchmark tests are run
// benchmark tests require valid path to VGG affine image/homography dataset
const bool RUN_UNIT_TESTS = false;
const bool RUN_BENCHMARK_TESTS = false;
const string VGG_ROOT_FOLDER_PATH = "C:\\Personal\\School\\UWB\\587\\FinalSource\\VGGImages"; // must be set to valid path if running benchmarks
// Number of feature matches to calculate for running benchmarks
const int BENCHMARK_MAX_FEATURES = 10000;
// Max distance between predicted target and actual target to consider valid match
const float BENCHMARK_EVAL_THRESHOLD = 3.0;


// Constants for window naming
const string NO_HGMS_WINDOW_NAME = "All Matches No HGMS";

void executeStagePipeline(string windowName, HGMSPipeline pipeline);

int main()
{
	// run unit tests and benchmark tests based on setting
	if (RUN_UNIT_TESTS)
	{
		HGMSUnitTests unitTests;
		unitTests.runUnitTests();
	}

	if (RUN_BENCHMARK_TESTS)
	{
		VGGBenchmarkRunner vggRunner;
		// run for all affine covariant folders
		cout << "Bark image dataset:" << std::endl;
		string barkFolderPath = VGG_ROOT_FOLDER_PATH + "\\bark";
		vggRunner.run(barkFolderPath, BENCHMARK_MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		vggRunner.printResults();

		cout << "Bikes image dataset:" << std::endl;
		string bikesFolderPath = VGG_ROOT_FOLDER_PATH + "\\bikes";
		vggRunner.run(bikesFolderPath, BENCHMARK_MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		vggRunner.printResults();

		cout << "Graf image dataset:" << std::endl;
		string grafFolderPath = VGG_ROOT_FOLDER_PATH + "\\graf";
		vggRunner.run(grafFolderPath, BENCHMARK_MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		vggRunner.printResults();

		cout << "Leuven image dataset:" << std::endl;
		string leuvenFolderPath = VGG_ROOT_FOLDER_PATH + "\\leuven";
		vggRunner.run(leuvenFolderPath, BENCHMARK_MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		vggRunner.printResults();

		cout << "Trees image dataset:" << std::endl;
		string treesFolderPath = VGG_ROOT_FOLDER_PATH + "\\trees";
		vggRunner.run(treesFolderPath, BENCHMARK_MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		vggRunner.printResults();

		cout << "UBC image dataset:" << std::endl;
		string ubcFolderPath = VGG_ROOT_FOLDER_PATH + "\\ubc";
		vggRunner.run(ubcFolderPath, BENCHMARK_MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		vggRunner.printResults();

		cout << "Wall image dataset:" << std::endl;
		string wallFolderPath = VGG_ROOT_FOLDER_PATH + "\\wall";
		vggRunner.run(wallFolderPath, BENCHMARK_MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		vggRunner.printResults();
	}

	// run demo of HGMS with LAT stage
	HGMSPipeline latPipeline(HGMSPipeline::AGGREGATE);
	latPipeline.addStage(std::make_shared<LATStage>());
	executeStagePipeline("LAT Filtered Matches", latPipeline);

	// run demo of HGMS with HGMS Stage
	HGMSPipeline hgmsPipeline(HGMSPipeline::AGGREGATE);
	hgmsPipeline.addStage(std::make_shared<HGMSStage>());
	executeStagePipeline("HGMS Filtered Matches", hgmsPipeline);

	// run demo of HGMS with MP Stage
	HGMSPipeline mpPipeline(HGMSPipeline::AGGREGATE);
	mpPipeline.addStage(std::make_shared<MPStage>());
	executeStagePipeline("MP Filtered Matches", mpPipeline);

	// run demo of HGMS with all stages registered
	HGMSPipeline allStagePipeline(HGMSPipeline::FILTER);
	allStagePipeline.addStage(std::make_shared<HGMSStage>());
	allStagePipeline.addStage(std::make_shared<LATStage>());
	allStagePipeline.addStage(std::make_shared<MPStage>());
	executeStagePipeline("All Stages Filtered Matches", allStagePipeline);

	return EXIT_SUCCESS;
}


void executeStagePipeline(string windowName, HGMSPipeline pipeline)
{
	// read images
	const Mat kittens1 = imread("kittens1.jpg", IMREAD_GRAYSCALE);
	const Mat kittens2 = imread("kittens2.jpg", IMREAD_GRAYSCALE);

	// create keypoints and descriptors
	std::vector<KeyPoint> k1kp, k2kp;
	Mat k1d, k2d;

	// create features
	Ptr<Feature2D> k1detector = ORB::create(10000);
	k1detector->detectAndCompute(kittens1, noArray(), k1kp, k1d);
	Ptr<Feature2D> k2detector = ORB::create(10000);
	k2detector->detectAndCompute(kittens2, noArray(), k2kp, k2d);

	// compute matches
	// Create brute-force matcher to compute matches between descriptors
	Ptr<BFMatcher> matcher = BFMatcher::create(cv::NORM_HAMMING, false);
	std::vector<DMatch> matchesAll;
	matcher->match(k1d, k2d, matchesAll);

	// execute stage with image parameters
	vector<DMatch> filteredMatches;
	pipeline.match(k1kp, kittens1.size(), k2kp, kittens2.size(), matchesAll, filteredMatches, 6.0f);

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// draw matches
	Mat filteredImageMatches;
	drawMatches(kittens1, k1kp, kittens2, k2kp, filteredMatches, filteredImageMatches);
	// resize and display matches
	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, filteredImageMatches.cols / 8, filteredImageMatches.rows / 8);
	imshow(windowName, filteredImageMatches);

	Mat noFilterMatches;
	drawMatches(kittens1, k1kp, kittens2, k2kp, matchesAll, noFilterMatches);
	// create resizeable window with scaling
	namedWindow(NO_HGMS_WINDOW_NAME, WINDOW_NORMAL);
	resizeWindow(NO_HGMS_WINDOW_NAME, noFilterMatches.cols / 8, noFilterMatches.rows / 8);
	imshow(NO_HGMS_WINDOW_NAME, noFilterMatches);

	// wait for input
	waitKey();

	// clean-up
	destroyAllWindows();

	// print metrics to screen
	cout << execMetrics;
}