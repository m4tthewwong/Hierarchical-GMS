//---------------------------------------------------------------------------  
// HGMSStage.cpp
// HGMS stage class file for representing hierarchical GMS processing stage. 
// This class derives from ProcessingStage and provide implement for hierarchical
// GMS feature match filtering. This class supports the following operations:
//	1. 
// Authors:  Pranshu Bhardwaj, Matthew Wong, Brennan O’Reilly
//---------------------------------------------------------------------------
// Inputs:
//  -- 
// 
// Outputs:
// -- 
// 
// Description:
//    This class provides the HGMS stage implementation.
//
// Assumptions:
//   -- This class must be instantiated before it can be added to the HGMSPipeline
// 

#pragma once

#include <string>
#include "HGMSStage.h"

/*----------------------------- default -----------------------------------
* Default constructor for class HGMSStage.
* Preconditions: Name of stage is required input parameter
* Postconditions: Instance of HGMSStage class is instantiated
*/
HGMSStage::HGMSStage() : ProcessingStage(STAGE_NAME)
{
}

/*----------------------------- destructor -------------------------------
* Destructor for class HGMSStage.
* Preconditions: Instance of HGMSStage class is instantiated.
* Postconditions: Objects are deallocated.
*/
HGMSStage::~HGMSStage()
{
}

/*----------------------------- execute -------------------------------
* Execute method for class HGMSStage.
* Preconditions: Instance of HGMSStage class is instantiated and image 1
*				 and 2 keypoints are provided, image1 and 2 sizes are provided,
*                image 1 and 2 feature matches are provided, thresholdFactor
*                is provided.
* Postconditions: The execute method is run and algorithm executed
*/
void HGMSStage::execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
	const std::vector<KeyPoint>& vkp2, const Size& size2,
	const std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
	const double thresholdFactor)
{
	// Call HGMS class execute method
	// create GMS matcher to computer matches between descriptors
	xfeatures2d::matchHGMS(size1, size2, vkp1, vkp2, matchesAll, vDMatches, false, false, thresholdFactor);
}