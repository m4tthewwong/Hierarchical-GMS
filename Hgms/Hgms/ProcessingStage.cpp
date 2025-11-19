//---------------------------------------------------------------------------  
// ProcessingStage.cpp
// Abstract class header file for representing processing stage. This class is
// meant to be derived and not instantiated by itself. This class supports the 
// following operations:
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
//    This class provides the stage abstract template class that represents a
//    processing stage in the HGMSPipeline.
//
// Assumptions:
//   -- This class must be derived by a concrete class to be utilized
// 
#pragma once

#include "ProcessingStage.h"

/*----------------------------- default -----------------------------------
* Default constructor for class ProcessingStage.
* Preconditions: none
* Postconditions: No instance of ProcessingStage class is instantiated as this
* class is a pure virtual class.
*/
ProcessingStage::ProcessingStage(const std::string name)
{
	stageName = name;
}

/*----------------------------- destructor -------------------------------
* Destructor for class ProcessingStage.
* Preconditions: Instance of ProcessingStage class is instantiated.
* Postconditions: Objects are deallocated.
*/
ProcessingStage::~ProcessingStage()
{
}

/*----------------------------- execute -------------------------------
* Execute method for class ProcessingStage.
* Preconditions: Instance of derived class from ProcessingStage class is 
*                instantiated and image 1 and 2 keypoints are provided, image1 and
*                2 sizes are provided, image 1 and 2 feature matches are provided,
*                thresholdFactor is provided.
* Postconditions: The derived stage object execute method is called
*/
void ProcessingStage::execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
	const std::vector<KeyPoint>& vkp2, const Size& size2, std::vector<DMatch>& matchesAll,
	std::vector<DMatch>& vDMatches, const double thresholdFactor)
{
}

/*----------------------------- getStageName -------------------------------
* Return name of stage.
* Preconditions: Instance of derived class from ProcessingStage is instantiated.
* Postconditions: Returns name of stage instantiated.
*/
const std::string ProcessingStage::getStageName() const
{
	return stageName;
}
