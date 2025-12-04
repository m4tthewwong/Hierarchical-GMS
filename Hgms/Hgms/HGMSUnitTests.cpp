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
#include "HGMSUnitTests.h"

using namespace std;

void HGMSUnitTests::runUnitTests()
{
	// validate stage registration
	validateHGMSPipeline();
	validateHGMSStageRegistration();
	validateLATStageRegistration();
	validateMPStageRegistration();
}

void HGMSUnitTests::validateHGMSPipeline()
{
	cout << "validateHGMSPipeline" << endl;

	// initialize pipeline
	HGMSPipeline pipeline(HGMSPipeline::AGGREGATE);

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// assert default values set
	assert(execMetrics.getPipelineMetrics().totalExecutionTimeMs == 0);

	// print metrics to screen
	cout << execMetrics;
}

void HGMSUnitTests::validateHGMSStageRegistration()
{
	cout << "validateHGMSStageRegistration" << endl;

	// initialize pipeline
	HGMSPipeline pipeline(HGMSPipeline::AGGREGATE);

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

void HGMSUnitTests::validateLATStageRegistration()
{
	cout << "validateLATStageRegistration" << endl;

	// initialize pipeline
	HGMSPipeline pipeline(HGMSPipeline::AGGREGATE);

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

void HGMSUnitTests::validateMPStageRegistration()
{
	cout << "validateMPStageRegistration" << endl;

	// initialize pipeline
	HGMSPipeline pipeline(HGMSPipeline::AGGREGATE);

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

std::vector<cv::KeyPoint> HGMSUnitTests::generateRandomKP(int num, int seed)
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

std::vector<cv::DMatch> HGMSUnitTests::generateMockMatches(int num)
{
	vector<DMatch> matches;

	for (int i = 0; i < num; i++)
	{
		matches.emplace_back(DMatch(i, i+1, 0, i + 0.5f));
	}

	return matches;
}

cv::Mat HGMSUnitTests::generateMockImages(int width, int height, int type)
{
	return Mat::zeros(height, width, type);
}
