//---------------------------------------------------------------------------  
// LATStage.h
// Local affine transformation stage header file for representing local affine GMS 
// processing stage. This class derives from ProcessingStage and provides 
// implementation for local affine GMS feature match filtering. This class 
// supports the following operations:
//	1. 
// Authors:  Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- 
// 
// Outputs:
// -- 
// 
// Description:
//    This class provides the local affine transformation stage implementation.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
//  

#pragma once

#include <string>
#include "LATStage.h"

/*----------------------------- default -----------------------------------
* Default constructor for class LATStage.
* Preconditions: Name of stage is required input parameter
* Postconditions: Instance of LATStage class is instantiated
*/
LATStage::LATStage() : ProcessingStage(STAGE_NAME)
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class LATStage.
* Preconditions: Instance of LATStage class is instantiated.
* Postconditions: Objects are deallocated.
*/
LATStage::~LATStage()
{
}

/*----------------------------- execute -------------------------------
* Execute method for class LATStage.
* Preconditions: Instance of LATStage class is instantiated and image 1
*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
*                image 1 and 2 feature matches are provided, thresholdFactor
*                is provided.
* Postconditions: The execute method is run and algorithm executed
*/
void LATStage::execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
	const std::vector<KeyPoint>& vkp2, const Size& size2,
	std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
	const double thresholdFactor)
{
}