//---------------------------------------------------------------------------  
// ProcessingStage.h
// Abstract stage header file for representing processing stage. This class is
// meant to be derived and not instantiated by itself. This class supports the 
// following operations:
//	1. Provides constructor/deconstructor
//  2. Provides execute method to run the stage
//  3. Provides ability to retrieve stage name
// Authors:  Pranshu Bhardwaj, Matthew Wong, Brennan O’Reilly
//---------------------------------------------------------------------------
// Inputs:
//  -- Stage must be constructed with a stageName parameter to identify the stage
// 
// Outputs:
// -- This class is not been to be instantiated directly but the derived classes
//    will be able to retrieve the stage name.
// 
// Description:
//    This class provides the stage abstract template class that represents a
//    processing stage in the HGMSPipeline.
//
// Assumptions:
//   -- This class must be derived by a concrete class to be utilized
// 

#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

using namespace cv;

class ProcessingStage {
public:
	/*----------------------------- default -----------------------------------
	* Default constructor for class ProcessingStage.
	* Preconditions: none
	* Postconditions: No instance of ProcessingStage class is instantiated as this
	* class is a pure virtual class.
	*/
	ProcessingStage(const std::string stageName);

	/*----------------------------- destructor -------------------------------
	* Destructor for class ProcessingStage.
	* Preconditions: Instance of ProcessingStage class is instantiated.
	* Postconditions: Objects are deallocated.
	*/
	virtual ~ProcessingStage();

	/*----------------------------- execute -------------------------------
	* Execute method for class ProcessingStage.
	* Preconditions: Instance of derived class from ProcessingStage class is
	*                instantiated and image 1 and 2 keypoints are provided, image1 and
	*                2 sizes are provided, image 1 and 2 feature matches are provided,
	*                thresholdFactor is provided.
	* Postconditions: The derived stage object execute method is called
	*/
	virtual void execute(const std::vector<KeyPoint>& vkp1, const Size& size1, 
		const std::vector<KeyPoint>& vkp2, const Size& size2, 
		const std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches, 
		const double thresholdFactor) = 0;

	/*----------------------------- getStageName -------------------------------
	* Return name of stage.
	* Preconditions: Instance of derived class from ProcessingStage is instantiated.
	* Postconditions: Returns name of stage instantiated.
	*/
	const std::string getStageName() const;


protected:
	// member variable for stage name
	std::string stageName;
};
