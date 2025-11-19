//-----------------------------------------------------------------------------  
// HGMSPipeline.h
// Orchestrator header file for registering/deregistering processing stages and
// retrieving performance metrics from execution. This class supports the 
// following operations:
//	1. 
// Authors:  Brennan O’Reilly, Matthew Wong, Pranshu Bhardwaj
//-----------------------------------------------------------------------------
// Inputs:
//  -- 
// 
// Outputs:
// -- 
// 
// Description:
//    This class provides the orchestrator or manager for the HGMS processing.
//    This class includes the ability to register pipeline stages that are classes
//    derviced from the ProcessingStage abstract class. Each pipeline stage
//    will return metrics that are combined and made available from a public method
//    in this class.
//
// Assumptions:
//   -- 
// 

#pragma once
#include "ProcessingStage.h"
#include "ExecutionMetrics.h"

class HGMSPipeline {
public:
	/*----------------------------- default -----------------------------------
	* Default constructor for class HGMSPipeline.
	* Preconditions: none.
	* Postconditions: Initialized HGMSPipeline object with data structures initialized
	*/
	HGMSPipeline();
	
	/*----------------------------- destructor --------------------------------
	* Destructor for class HGMSPipeline.
	* Preconditions: none.
	* Postconditions: This method will perform clean-up and deallocation
	*/
	~HGMSPipeline();
	
	/*----------------------------- addStage ----------------------------------
	* Method to add initialized stage object to collection of stages to run
	* Preconditions: Initialized stage object derived from ProcessingStage
	* Postconditions: This method will return boolean indicating true if stage
	*                 was added to collection or false if it was not added.
	*/
	const bool addStage(const std::shared_ptr<ProcessingStage> stage);

	/*----------------------------- deleteStage -------------------------------
	* Method to delete registered stage
	* Preconditions: Name of stage to remove
	* Postconditions: This method will remove the stage from the collection if
	*                 the stage exists and return a true or false if the stage
	*                 was deleted.
	*/
	const bool deleteStage(const std::string stageName);

	/*----------------------------- getStageSize ------------------------------
	* Method to return the number of registered stages in pipeline
	* Preconditions: Initialized pipeline object
	* Postconditions: The number of registered pipeline stages
	*/
	const int getStageSize();

	/*----------------------------- match -------------------------------------
	* Method to execute stage processing
	* Preconditions: Keypoints for image1 and image2, size of image1 and image2,
	*                detected feature matches, and threshold
	* Postconditions: This method will perform the execution of all registered stages
	*/
	void match(const std::vector<KeyPoint>& vkp1, const Size& size1, 
		const std::vector<KeyPoint>& vkp2, const Size& size2, 
		std::vector<DMatch>& matchesAll, std::vector<DMatch>& vDMatches, 
		const double thresholdFactor);

	/*----------------------------- getExecMetrics ----------------------------
	* Method to execute stage processing
	* Preconditions: Keypoints for image1 and image2, size of image1 and image2,
	*                detected feature matches, and threshold
	* Postconditions: This method will perform the execution of all registered stages
	*/
	const ExecutionMetrics getExecMetrics();

private:
	double mThresholdFactor = 0;

	// Data structure that contains all registered stages
	std::vector<std::shared_ptr<ProcessingStage>> pipelineStages;

	// Instance of pipeline execution metrics
	ExecutionMetrics pipeExecMetrics;
};