//---------------------------------------------------------------------------  
// HGMSStage.h
// HGMS stage header file for representing hierarchical GMS processing stage. 
// This class derives from ProcessingStage and provides implementation for hierarchical
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
#include "ProcessingStage.h"
#include "hgms.hpp"

class HGMSStage : public ProcessingStage {
public:
	/*----------------------------- default -----------------------------------
	* Default constructor for class HGMSStage.
	* Preconditions: none
	* Postconditions: Instance of HGMSStage class is instantiated
	*/
	HGMSStage();

	/*----------------------------- destructor -------------------------------
	* Destructor for class HGMSStage.
	* Preconditions: Instance of HGMSStage class is instantiated.
	* Postconditions: Objects are deallocated.
	*/
	virtual ~HGMSStage();

	/*----------------------------- execute -------------------------------
	* Execute method for class HGMSStage.
	* Preconditions: Instance of HGMSStage class is instantiated and image 1 
	*				 and 2 keypoints are provided, image1 and 2 sizes are provided, 
	*                image 1 and 2 feature matches are provided, thresholdFactor 
	*                is provided.
	* Postconditions: The execute method is run and algorithm executed
	*/
	virtual void execute(const std::vector<KeyPoint>& vkp1, const Size& size1,
		const std::vector<KeyPoint>& vkp2, const Size& size2,
		std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches,
		const double thresholdFactor);

private:
	// class identifier 
	static constexpr const char* STAGE_NAME = "HGMS";

};
