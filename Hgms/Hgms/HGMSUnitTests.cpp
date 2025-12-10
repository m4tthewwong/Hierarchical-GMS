//---------------------------------------------------------------------------
// HGMSUnitTest.cpp
// Driver code for testing the HGMS algorithm and associated class files
// Performs the following operations:
//	1. Unit tests of pipeline stage registration
//  2. Unit tests of metrics output
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

/*----------------------------- default -----------------------------------
* Default constructor for class HGMSUnitTests.
* Preconditions: none
* Postconditions: Instance of HGMSUnitTests class is instantiated
*/
HGMSUnitTests::HGMSUnitTests()
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class 	~HGMSUnitTests();.
* Preconditions: Instance of ~HGMSUnitTests() class is instantiated.
* Postconditions: Objects are deallocated.
*/
HGMSUnitTests::~HGMSUnitTests()
{
}

/*----------------------------- runUnitTests -------------------------------
* Public runUnitTests method which runs the unit tests.
* Preconditions: Instance of HGMSUnitTests instantiated
* Postconditions: Unit tests for testing pipeline registration and printing of
*                 metrics executed.
*/
void HGMSUnitTests::runUnitTests()
{
	// validate stage registration
	validateHGMSPipeline();
	validateHGMSStageRegistration();
	validateLATStageRegistration();
	validateMPStageRegistration();
	validateNoStageHGMSPipeline();
}

/*----------------------------- validateHGMSPipeline --------------------------
* Private helper for validating the registration of HGMS pipeline
* Preconditions: Initialized HGMSUnitTests class
* Postconditions: assert if result does not match expected output
*/
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

/*------------------- validateHGMSStageRegistration -----------------------
* Private helper for validating the registration of HGMS stage
* Preconditions: Initialized HGMSUnitTests class
* Postconditions: assert if result does not match expected output
*/
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

/*------------------- validateLATStageRegistration -----------------------
* Private helper for validating the registration of LAT stage
* Preconditions: Initialized HGMSUnitTests class
* Postconditions: assert if result does not match expected output
*/
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

/*------------------- validateMPStageRegistration -----------------------
* Private helper for validating the registration of MP stage
* Preconditions: Initialized HGMSUnitTests class
* Postconditions: assert if result does not match expected output
*/
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

/*------------------- validateNoStageHGMSPipeline -----------------------
* Private helper for validating the behavior of no stage registration
* Preconditions: Initialized HGMSUnitTests class
* Postconditions: assert if result does not match expected output
*/
void HGMSUnitTests::validateNoStageHGMSPipeline()
{
	cout << "validateNoStageHGMSPipeline" << endl;

	// initialize pipeline
	HGMSPipeline pipeline(HGMSPipeline::AGGREGATE);

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// print metrics to screen
	cout << execMetrics;
}

/*----------------------------- generateRandomKP -------------------------------
* Private helper generateRandomKP method for generating mock key point data.
* Preconditions: non-zero num and seed are provided
* Postconditions: Vector of KeyPoints generated
*/
std::vector<cv::KeyPoint> HGMSUnitTests::generateRandomKP(int num, int seed)
{
	vector<KeyPoint> keypoints;
	if (num <= 0 || seed <= 0)
	{
		return keypoints;
	}

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

/*----------------------------- generateMockMatches -------------------------------
* Private helper generateMockMatches method for generating mock DMatch data.
* Preconditions: non-zero num is provided
* Postconditions: Vector of DMatch generated
*/
std::vector<cv::DMatch> HGMSUnitTests::generateMockMatches(int num)
{
	vector<DMatch> matches;

	for (int i = 0; i < num; i++)
	{
		matches.emplace_back(DMatch(i, i+1, 0, i + 0.5f));
	}

	return matches;
}

/*----------------------------- generateMockImages -------------------------------
* Private helper generateMockImages method for generating mock image data.
* Preconditions: non-zero width, height, and type are provided
* Postconditions: Mat generated of sample image
*/
cv::Mat HGMSUnitTests::generateMockImages(int width, int height, int type)
{
	if (width <= 0 || height <= 0 || type <= 0)
	{
		return Mat();
	}

	return Mat::zeros(height, width, type);
}
