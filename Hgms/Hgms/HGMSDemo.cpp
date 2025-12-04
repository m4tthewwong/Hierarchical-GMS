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

// Constants for window naming
const string NO_HGMS_WINDOW_NAME = "All Matches No HGMS";

void executeStagePipeline(string windowName, HGMSPipeline pipeline);

int main()
{
	//// run unit tests
	//HGMSUnitTests unitTests;
	//unitTests.runUnitTests();

	//// run demo of HGMS with LAT stage
	//HGMSPipeline latPipeline(HGMSPipeline::AGGREGATE);
	//latPipeline.addStage(std::make_shared<LATStage>());
	//executeStagePipeline("LAT Filtered Matches", latPipeline);

	//// run demo of HGMS with HGMS Stage
	//HGMSPipeline hgmsPipeline(HGMSPipeline::AGGREGATE);
	//hgmsPipeline.addStage(std::make_shared<HGMSStage>());
	//executeStagePipeline("HGMS Filtered Matches", hgmsPipeline);

	//// run demo of HGMS with MP Stage
	//HGMSPipeline mpPipeline(HGMSPipeline::AGGREGATE);
	//mpPipeline.addStage(std::make_shared<MPStage>());
	//executeStagePipeline("MP Filtered Matches", mpPipeline);

	// run demo of HGMS with all stages registered
	HGMSPipeline allStagePipeline(HGMSPipeline::FILTER);
	allStagePipeline.addStage(std::make_shared<LATStage>());
	allStagePipeline.addStage(std::make_shared<HGMSStage>());
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
	Ptr<BFMatcher> matcher = BFMatcher::create();
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