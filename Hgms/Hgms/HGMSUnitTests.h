
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
	/*----------------------------- default -----------------------------------
	* Default constructor for class HGMSUnitTests.
	* Preconditions: none
	* Postconditions: Instance of HGMSUnitTests class is instantiated
	*/
	HGMSUnitTests();

	/*----------------------------- destructor -------------------------------
	* Destructor for class 	~HGMSUnitTests();.
	* Preconditions: Instance of ~HGMSUnitTests() class is instantiated.
	* Postconditions: Objects are deallocated.
	*/
	~HGMSUnitTests();

	/*----------------------------- runUnitTests -------------------------------
	* Public runUnitTests method which runs the unit tests.
	* Preconditions: Instance of HGMSUnitTests instantiated
	* Postconditions: Unit tests for testing pipeline registration and printing of
	*                 metrics executed.
	*/
	void runUnitTests();

private:
	const int RAND_SEED = 42;

	/*------------------------ generateRandomKP -------------------------------
	* Private helper generateRandomKP method for generating mock key point data.
	* Preconditions: non-zero num and seed are provided
	* Postconditions: Vector of KeyPoints generated
	*/
	std::vector<cv::KeyPoint> generateRandomKP(int num, int seed);

	/*--------------------- generateMockMatches -------------------------------
	* Private helper generateMockMatches method for generating mock DMatch data.
	* Preconditions: non-zero num is provided
	* Postconditions: Vector of DMatch generated
	*/
	std::vector<cv::DMatch> generateMockMatches(int num);

	/*---------------------- generateMockImages -------------------------------
	* Private helper generateMockImages method for generating mock image data.
	* Preconditions: non-zero width, height, and type are provided
	* Postconditions: Mat generated of sample image
	*/	
	cv::Mat generateMockImages(int width, int height, int type = CV_8UC1);

	/*-------------------- validateHGMSPipeline -------------------------------
	* Private helper for validating the registration of HGMS pipeline
	* Preconditions: Initialized HGMSUnitTests class
	* Postconditions: assert if result does not match expected output
	*/
	void validateHGMSPipeline();
	
	/*------------------- validateHGMSStageRegistration -----------------------
	* Private helper for validating the registration of HGMS stage
	* Preconditions: Initialized HGMSUnitTests class
	* Postconditions: assert if result does not match expected output
	*/
	void validateHGMSStageRegistration();

	/*------------------- validateLATStageRegistration -----------------------
	* Private helper for validating the registration of LAT stage
	* Preconditions: Initialized HGMSUnitTests class
	* Postconditions: assert if result does not match expected output
	*/
	void validateLATStageRegistration();

	/*------------------- validateMPStageRegistration -----------------------
	* Private helper for validating the registration of MP stage
	* Preconditions: Initialized HGMSUnitTests class
	* Postconditions: assert if result does not match expected output
	*/
	void validateMPStageRegistration();

	/*------------------- validateNoStageHGMSPipeline -----------------------
	* Private helper for validating the behavior of no stage registration
	* Preconditions: Initialized HGMSUnitTests class
	* Postconditions: assert if result does not match expected output
	*/
	void validateNoStageHGMSPipeline();;

};