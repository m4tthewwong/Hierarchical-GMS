//---------------------------------------------------------------------------  
// HGMSPipeline.cpp
// Orchestrator class file for registering/deregistering processing stages, 
// stage executions, and retrieving performance metrics from execution. This 
// class supports the following operations:
//	1. 
// Authors:  Brennan O’Reilly, Matthew Wong, Pranshu Bhardwaj
//---------------------------------------------------------------------------
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

#include "HGMSPipeline.h"

/*----------------------------- default ---------------------------------------
* Default constructor for class HGMSPipeline.
* Preconditions: Processing mode must be provided as input to constructor
* Postconditions: Initialized HGMSPipeline object with data structures initialized
*/
HGMSPipeline::HGMSPipeline(ProcessingMode mode) : processingMode(mode)
{
}

/*----------------------------- destructor ------------------------------------
* Destructor for class HGMSPipeline.
* Preconditions: none.
* Postconditions: This method will perform clean-up and deallocation
*/
HGMSPipeline::~HGMSPipeline()
{
}

/*----------------------------- addStage -----------------------
* Method to add initialized stage object to collection of stages to run
* Preconditions: Initialized stage object derived from ProcessingStage with unique stage name
* Postconditions: This method will return boolean indicating true if stage
*                 was added to collection or false if it was not added. If stage
*                 name is not unique then the stage will be deleted and replaced.
*/
const bool HGMSPipeline::addStage(const std::shared_ptr<ProcessingStage> stage)
{
	// check that stage is not null
	if (stage)
	{
		bool replaced = false;

		// Check pipeline name and replace if it exists otherwise add to collection
		for (auto& pipeStage : pipelineStages)
		{
			if (pipeStage->getStageName() == stage->getStageName())
			{
				pipeStage = stage;
				replaced = true;
			}
		}

		// If replaced is false and it is a valid stage, then add
		if (replaced == false)
		{
			pipelineStages.push_back(stage);
		}
		return true;
	}

	// unable to add to collection
	return false;
}

/*----------------------------- deleteStage -----------------------
* Method to delete registered stage
* Preconditions: Name of stage to remove
* Postconditions: This method will remove the stage from the collection if
*                 the stage exists and return a true or false if the stage
*                 was deleted.
*/
const bool HGMSPipeline::deleteStage(const std::string stageName)
{
	// Check pipeline name and replace if it exists otherwise add to collection
	for (auto itr = pipelineStages.begin(); itr != pipelineStages.end(); ++itr)
	{
		if ((*itr)->getStageName() == stageName)
		{
			pipelineStages.erase(itr);
			return true;
		}
	}

	// unable to remove from collection
	return false;
}

/*----------------------------- getStageSize ------------------------------
* Method to return the number of registered stages in pipeline
* Preconditions: Initialized pipeline object
* Postconditions: The number of registered pipeline stages
*/
const int HGMSPipeline::getStageSize()
{
	return pipelineStages.size();
}

/*----------------------------- match -----------------------
* Method to execute stage processing
* Preconditions: Keypoints for image1 and image2, size of image1 and image2,
*                detected feature matches, and threshold
* Postconditions: This method will perform the execution of all registered stages
*                 and save execution metrics
*/
void HGMSPipeline::match(const std::vector<KeyPoint>& vkp1, const Size& size1,
	const std::vector<KeyPoint>& vkp2, const Size& size2, const std::vector<DMatch>& matchesAll,
	std::vector<DMatch>& vDMatches, const double thresholdFactor)
{
	// Initialize the pipeExecMetrics object
	pipeExecMetrics.initialize();

	// allocate size of vDMatches to max matches * number of stages
	vDMatches.reserve(matchesAll.size() * pipelineStages.size());

	// create copy of input Matches
	std::vector<DMatch> inputMatches(matchesAll);

	// Setup timers
	for (auto& stage : pipelineStages)
	{
		std::vector<DMatch> stageMatches(matchesAll.size());
		
		if (stage)
		{
			// initialize time and execute stage
			TickMeter tm;
			tm.start();

			// execute stage
			stage->execute(vkp1, size1, vkp2, size2, inputMatches, stageMatches, thresholdFactor);
			
			// stop timer
			tm.stop();
			
			// populate stage metrics and add to collection
			ExecutionMetrics::StageExecMetrics stageMetrics;
			stageMetrics.stageName = stage->getStageName();
			stageMetrics.stageMatchesInputSize = matchesAll.size();
			stageMetrics.stageMatchesOutputSize = stageMatches.size();
			stageMetrics.executionTimeMs = tm.getTimeMilli();

			// add execution metrics to collection
			pipeExecMetrics.addStageMetrics(stageMetrics);

			// increment total pipeline metrics
			ExecutionMetrics::PipelineExecMetrics pipeMetrics;
			pipeMetrics.totalExecutionTimeMs = tm.getTimeMilli();
			pipeMetrics.totalRawMatchesInputSize = matchesAll.size();
			pipeMetrics.totalUniqueMatchesOutputSize = vDMatches.size();
			pipeExecMetrics.updatePipelineExecMetrics(pipeMetrics);
		}

		// Insert matches from stage
		vDMatches.insert(vDMatches.end(), stageMatches.begin(), stageMatches.end());

		// Reset inputMatches if processing mode set to filter
		if (processingMode == FILTER)
		{
			inputMatches = stageMatches;
		}
	}
}

/*----------------------------- getExecMetrics ----------------------------
* Method to execute stage processing
* Preconditions: Keypoints for image1 and image2, size of image1 and image2,
*                detected feature matches, and threshold
* Postconditions: This method will perform the execution of all registered stages
*/
const ExecutionMetrics HGMSPipeline::getExecMetrics()
{
	return pipeExecMetrics;
}
