//---------------------------------------------------------------------------
// HGMSUnitTest.cpp
// Driver code for testing the HGMS algorithm and associated class files
// Performs the following operations:
//	1. 
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
#include "HGMSPipeline.h"
#include "HGMSStage.h"
#include "LATStage.h"
#include "MPStage.h"

using namespace std;

const int RAND_SEED = 42;

void runUnitTests();

// Utility method for generating mock key point data
std::vector<cv::KeyPoint> generateRandomKP(int num, int seed);

// Utility method for generating mock matches data
std::vector<cv::DMatch> generateMockMatches(int num);

// Utility method for generating mock image data
cv::Mat generateMockImages(int width, int height, int type = CV_8UC1);

// validation for initialization
void validateHGMSPipeline();
void validateHGMSStageRegistration();
void validateLATStageRegistration();
void validateHGMSStageRegistration();
void validateMPStageRegistration();

// validation for execution using mock data
void validateAllStageExecution();


int main()
{
	cout << "Executing HGMS Unit Tests" << endl;
	runUnitTests();

	return EXIT_SUCCESS;
}

void runUnitTests()
{
	// validate stage registration
	validateHGMSPipeline();
	validateHGMSStageRegistration();
	validateLATStageRegistration();
	validateMPStageRegistration();

	// validate execution
	validateAllStageExecution();

}

void validateHGMSPipeline()
{
	cout << "validateHGMSPipeline" << endl;

	// initialize pipeline
	HGMSPipeline pipeline;

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// assert default values set
	assert(execMetrics.getPipelineMetrics().totalExecutionTimeMs == 0);

	// print metrics to screen
	cout << execMetrics;
}

void validateHGMSStageRegistration()
{
	cout << "validateHGMSStageRegistration" << endl;

	// initialize pipeline
	HGMSPipeline pipeline;

	// add hgms stage
	pipeline.addStage(std::make_shared<HGMSStage>());

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// assert default values set
	assert(execMetrics.getPipelineMetrics().totalExecutionTimeMs == 0);
	assert(pipeline.getStageSize() == 1);

	// print metrics to screen
	cout << execMetrics;
}

void validateLATStageRegistration()
{
	cout << "validateLATStageRegistration" << endl;

	// initialize pipeline
	HGMSPipeline pipeline;

	// add hgms stage
	pipeline.addStage(std::make_shared<LATStage>());

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// assert default values set
	assert(execMetrics.getPipelineMetrics().totalExecutionTimeMs == 0);
	assert(pipeline.getStageSize() == 1);

	// print metrics to screen
	cout << execMetrics;
}

void validateMPStageRegistration()
{
	cout << "validateMPStageRegistration" << endl;

	// initialize pipeline
	HGMSPipeline pipeline;

	// add hgms stage
	pipeline.addStage(std::make_shared<MPStage>());

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// assert default values set
	assert(execMetrics.getPipelineMetrics().totalExecutionTimeMs == 0);
	assert(pipeline.getStageSize() == 1);

	// print metrics to screen
	cout << execMetrics;
}

void validateAllStageExecution()
{
	cout << "validateAllStageExecution" << endl;
	int dataPointNum = 5;

	// generate mock keypoints
	auto kp1 = generateRandomKP(dataPointNum, RAND_SEED);
	auto kp2 = generateRandomKP(dataPointNum, RAND_SEED);

	// generate mats
	cv::Mat img1 = generateMockImages(10, 10, CV_8UC1);
	cv::Mat img2 = generateMockImages(10, 10, CV_8UC1);
	
	// generate mock matches
	auto matchesAll = generateMockMatches(dataPointNum);

	// initialize pipeline and HGMSStage
	HGMSPipeline pipeline;
	pipeline.addStage(std::make_shared<HGMSStage>());
	pipeline.addStage(std::make_shared<LATStage>());
	pipeline.addStage(std::make_shared<MPStage>());

	// execute stage with mock parameters
	vector<DMatch> filteredMatches;
	pipeline.match(kp1, img1.size(), kp2, img2.size(), matchesAll, filteredMatches, 6.0f);

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();
	
	// assert default values set
	assert(execMetrics.getPipelineMetrics().totalExecutionTimeMs > 0);
	assert(pipeline.getStageSize() == 3);

	// print metrics to screen
	cout << execMetrics;
}

std::vector<cv::KeyPoint> generateRandomKP(int num, int seed)
{
	vector<KeyPoint> keypoints;

	for (int i = 0; i < num; i++)
	{
		// standard mersenne_twister_engine seeded with seed
		// create random uniform distribution of numbers for mock keypoints
		std::mt19937 gen(seed);
		std::uniform_real_distribution<float> urdis(0.f, 100.f);
		keypoints.emplace_back(KeyPoint(urdis(gen), urdis(gen), 5.f));
	}

	return keypoints;
}

std::vector<cv::DMatch> generateMockMatches(int num)
{
	vector<DMatch> matches;

	for (int i = 0; i < num; i++)
	{
		matches.emplace_back(DMatch(i, i+1, 0, i + 0.5f));
	}

	return matches;
}

cv::Mat generateMockImages(int width, int height, int type)
{
	return Mat::zeros(height, width, type);
}
