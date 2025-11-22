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

#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "HGMSUnitTests.h"
#include "HGMSPipeline.h"
#include "HGMSStage.h"
#include "LATStage.h"
#include "MPStage.h"

void executeHGMSUnitTests();
void executeHGMSLatPipeline();

int main()
{
	// run unit tests
	//executeHGMSUnitTests();

	// run demo of HGMS with LAT stage
	executeHGMSLatPipeline();
	
	return EXIT_SUCCESS;
}

void executeUnitTests()
{
	HGMSUnitTests unitTests;
	unitTests.runUnitTests();

}

void executeHGMSLatPipeline()
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

	// initialize pipeline and HGMSStage
	HGMSPipeline pipeline;
	pipeline.addStage(std::make_shared<LATStage>());

	// execute stage with mock parameters
	vector<DMatch> filteredMatches;
	pipeline.match(k1kp, kittens1.size(), k2kp, kittens2.size(), matchesAll, filteredMatches, 6.0f);

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// draw matches
	Mat filteredImageMatches;
	drawMatches(kittens1, k1kp, kittens2, k2kp, filteredMatches, filteredImageMatches);
	// resize and display matches
	namedWindow("LAT Filtered Matches", WINDOW_NORMAL);
	resizeWindow("LAT Filtered Matches", filteredImageMatches.cols / 8, filteredImageMatches.rows / 8);
	imshow("LAT Filtered Matches", filteredImageMatches);

	Mat noFilterMatches;
	drawMatches(kittens1, k1kp, kittens2, k2kp, matchesAll, noFilterMatches);
	// create resizeable window with scaling
	namedWindow("Orig All Matches", WINDOW_NORMAL);
	resizeWindow("Orig All Matches", noFilterMatches.cols / 8, noFilterMatches.rows / 8);
	imshow("Orig All Matches", noFilterMatches);


	// wait for input
	waitKey();

	// clean-up
	destroyAllWindows();

	// print metrics to screen
	cout << execMetrics;
}