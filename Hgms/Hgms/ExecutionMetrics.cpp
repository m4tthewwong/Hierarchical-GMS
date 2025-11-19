//---------------------------------------------------------------------------  
// ExecutionMetrics.h
// Metrics class file for generating metrics/statistics from each stage execution.
// This class provides statistics for each execution stage so that the performance
// can be evaluated and compared against other algorithms. This class supports 
// the following operations:
//	1. 
// Authors:  Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- StageExecMetrics which represents the metric for one execution stage
// 
// Outputs:
// -- Collection of StageExecMetrics data structures
// -- PipelineExecMetrics data structure
// 
// Description:
//    This class provides methods to profile execution performance for each
//    pipeline processing stage.
//
// Assumptions:
//   -- 
// 
#pragma once
#include <sstream>
#include "ExecutionMetrics.h"

using namespace std;

/*----------------------------- default ---------------------------------------
* Default constructor for class ExecutionMetrics.
* Preconditions: none.
* Postconditions: Initialized ExecutionMetrics object with data structures initialized
*/
ExecutionMetrics::ExecutionMetrics()
{
	initialize();
}

/*----------------------------- destructor ------------------------------------
* Destructor for class ExecutionMetrics.
* Preconditions: none.
* Postconditions: This method will perform clean-up and deallocation
*/
ExecutionMetrics::~ExecutionMetrics()
{
}

/*------------------------- initialize ------------------------------------
* Method to initialize the pipeline and stage metrics
* Preconditions: none.
* Postconditions: This method will initialize the pipelineExecMetric to
*                 default values and remove any prior stageExecMetrics.
*/
void ExecutionMetrics::initialize()
{
	// Initialize the pipeline metrics
	pipelineExecMetrics.totalRawMatchesInputSize = 0;
	pipelineExecMetrics.totalUniqueMatchesOutputSize = 0;
	pipelineExecMetrics.totalExecutionTimeMs = 0;

	// clear stages collection of any initialized stages
	stagesExecMetrics.clear();
}

/*----------------------------- getAllStagesMetrics ---------------------------
* Method to return collection of all stages metrics
* Preconditions: Collection of initialized stageExecMetric data structures
* Postconditions: This method will perform clean-up and deallocation
*/
const std::vector<ExecutionMetrics::StageExecMetrics>& 
	ExecutionMetrics::getAllStagesMetrics() const
{
	return stagesExecMetrics;
}

/*----------------------------- getPipelineMetrics ----------------------------
* Method to return collection of pipeline metrics
* Preconditions: Initialized pipelineExecMetrics data structure
* Postconditions: This method will return the pipelineExecMetrics data struct
*/
const ExecutionMetrics::PipelineExecMetrics& 
	ExecutionMetrics::getPipelineMetrics() const
{
	return pipelineExecMetrics;
}

/*----------------------------- addStageMetrics -------------------------------
* Method to add stage execution metrics to collection
* Preconditions: Initialized stageMetric object.
* Postconditions: This method will add stageMetric to collection
*/
void ExecutionMetrics::addStageMetrics(const StageExecMetrics& stageMetrics)
{
	if (!stageMetrics.stageName.empty()) {
		stagesExecMetrics.push_back(stageMetrics);
	}
}

/*-----------------------------  print  -----------------------------------
* Overloaded output operator for class ExecutionMetrics.
* Preconditions: None.
* Postconditions: The value of pipeline metrics and stage metrics is sent to
*                 the output stream.
*/
void ExecutionMetrics::print(std::ostream& outputStream) const
{
	outputStream << "Total Pipeline Exec Time: " << pipelineExecMetrics.totalExecutionTimeMs << "\t"
	<< "Total Match Inputs: " << pipelineExecMetrics.totalRawMatchesInputSize << "\t"
	<< "Total Unique Outputs: " << pipelineExecMetrics.totalUniqueMatchesOutputSize << endl;

	for (auto& stageMetric : stagesExecMetrics)
	{
		outputStream << "---------------------------------------------" << endl;
		outputStream << "Stage: " << stageMetric.stageName << "\t"
			<< "Stage Pipeline Exec Time: " << stageMetric.executionTimeMs << "\t"
			<< "Stage Match Inputs: " << stageMetric.stageMatchesInputSize << "\t"
			<< "Stage Unique Outputs: " << stageMetric.stageMatchesOutputSize << endl;
	}
}

/*----------------------------- updatePipelineExecMetrics ---------------------
* Method to update total pipeline execution metrics
* Preconditions: Initialized pipelineMetrics data structure
* Postconditions: This method will update the pipelineExecMetrics data struct
*/
void ExecutionMetrics::updatePipelineExecMetrics
	(const ExecutionMetrics::PipelineExecMetrics& pipelineMetrics)
{
	pipelineExecMetrics.totalRawMatchesInputSize += pipelineMetrics.totalRawMatchesInputSize;
	pipelineExecMetrics.totalUniqueMatchesOutputSize += pipelineMetrics.totalUniqueMatchesOutputSize;
	pipelineExecMetrics.totalExecutionTimeMs += pipelineMetrics.totalExecutionTimeMs;
}

/*-----------------------------  <<  ------------------------------------------
* Overloaded output operator for class SportsCard.
* Preconditions: None.
* Postconditions: This function is to allow of printing using the classes
*                 print command.
*/
std::ostream& operator<<(std::ostream& outputStream, const ExecutionMetrics& rhs)
{
	rhs.print(outputStream);
	return outputStream;
}
