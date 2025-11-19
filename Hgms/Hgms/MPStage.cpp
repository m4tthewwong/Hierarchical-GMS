//---------------------------------------------------------------------------  
// MPStage.cpp
// Multi-plane stage header file for representing multi-plane GMS processing stage. 
// This class derives from ProcessingStage and provide implement for multi-plane
// GMS feature match filtering. This class supports the following operations:
//	1. 
// Authors:  Matthew Wong, Brennan O’Reilly, Pranshu Bhardwaj
//---------------------------------------------------------------------------
// Inputs:
//  -- 
// 
// Outputs:
// -- 
// 
// Description:
//    This class provides the Multi-plane stage implementation.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
// 

#pragma once

#include <string>
#include "MPStage.h"

/*----------------------------- default -----------------------------------
* Default constructor for class MPStage.
* Preconditions: Name of stage is required input parameter
* Postconditions: Instance of MPStage class is instantiated
*/
MPStage::MPStage() : ProcessingStage(STAGE_NAME)
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class MPStage.
* Preconditions: Instance of MPStage class is instantiated.
* Postconditions: Objects are deallocated.
*/
MPStage::~MPStage()
{
}

/*----------------------------- execute -------------------------------
* Execute method for class MPStage.
* Preconditions: Instance of MPStage class is instantiated and image 1
*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
*                image 1 and 2 feature matches are provided, thresholdFactor
*                is provided.
* Postconditions: The execute method is run and algorithm executed
*/
void MPStage::execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
	const std::vector<KeyPoint>& vkp2, const Size& size2,
	std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
	const double thresholdFactor)
{
}