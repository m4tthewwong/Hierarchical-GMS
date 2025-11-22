
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

class HGMSUnitTests 
{
public:
	void runUnitTests();

private:
	const int RAND_SEED = 42;

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
	void validateMPStageRegistration();

	// validation for execution using mock data
	void validateAllStageExecution();

};