//---------------------------------------------------------------------------  
// ExecutionMetrics.h
// Metrics header file for generating metrics/statistics from each stage execution.
// This class provides statistics for each execution stage so that the performance
// can be evaluated and compared against other algorithms. This class supports 
// the following operations:
//	1. Storage of matches discovered per stage
//  2. Print method to output metrics
// Authors:  Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- StageExecMetrics which represents the metric for one execution stage
// 
// Outputs:
//  -- Collection of StageExecMetrics data structures
//  -- PipelineExecMetrics data structure
// 
// Description:
//    This class provides methods to profile execution performance for each
//    pipeline processing stage.
//
// Assumptions:
//   -- This class is meant to be called from the HGMSPipeline class and any other
//      class that requires output of performance from HGMSPipeline.
// 

#pragma once

#include <string>
#include <vector>

class ExecutionMetrics {
	/*-----------------------------  <<  --------------------------------------
	* Overloaded output operator for class ExecutionMetrics.
	* Preconditions: None.
	* Postconditions: This function is to allow of printing using the classes
	*                 print command.
	*/
	friend std::ostream& operator<<(std::ostream& outputStream,
		const ExecutionMetrics& rhs);

public:
	/*----------------------------- default ---------------------------------------
	* Default constructor for class ExecutionMetrics.
	* Preconditions: none.
	* Postconditions: Initialized ExecutionMetrics object with data structures initialized
	*/
	ExecutionMetrics();
	
	/*----------------------------- destructor --------------------------------
	* Destructor for class ExecutionMetrics.
	* Preconditions: Initialized ExecutionMetrics class that is being dereferenced
	* Postconditions: This method will perform clean-up and deallocation
	*/
	~ExecutionMetrics();

	// data structure to hold metrics about stage execution
	struct StageExecMetrics {
		std::string stageName = "";
		size_t stageMatchesInputSize = 0;
		size_t stageMatchesOutputSize = 0;
		double executionTimeMs = 0;
	};

	// data structure hold pipeline execution metrics across all stages
	struct PipelineExecMetrics {
		size_t totalRawMatchesInputSize = 0;
		size_t totalUniqueMatchesOutputSize = 0;
		double totalExecutionTimeMs = 0;
	};

	/*----------------------------- getAllStagesMetrics -----------------------
	* Method to return collection of all stages metrics
	* Preconditions: Collection of initialized stageExecMetric data structures
	* Postconditions: This method will perform clean-up and deallocation
	*/
	const std::vector<StageExecMetrics>& getAllStagesMetrics() const;

	/*----------------------------- getPipelineMetrics ------------------------
	* Method to return collection of pipeline metrics
	* Preconditions: Initialized pipelineExecMetrics data structure
	* Postconditions: This method will return the pipelineExecMetrics data struct
	*/
	const PipelineExecMetrics& getPipelineMetrics() const;

	/*----------------------------- addStageMetrics ---------------------------
	* Method to add stage execution metrics to collection
	* Preconditions: Initialized stageMetric object.
	* Postconditions: This method will add stageMetric to collection
	*/
	void addStageMetrics(const StageExecMetrics& stageMetrics);

	/*-----------------------------  print  -----------------------------------
	* Overloaded output operator for class ExecutionMetrics.
	* Preconditions: Instance of ExecutionMetrics has been instantiated and populated
	* with metric.
	* Postconditions: The output of the pipeline execution metrics and metrics from
	* each stage.
	*/
	void print(std::ostream& outputStream) const;

	/*------------------------- updatePipelineExecMetrics ---------------------
	* Method to update total pipeline execution metrics
	* Preconditions: Initialized pipelineMetrics data structure
	* Postconditions: This method will update the pipelineExecMetrics data struct
	*/	
	void updatePipelineExecMetrics(const PipelineExecMetrics& pipelineMetrics);

	/*------------------------- initialize ------------------------------------
	* Method to initialize the pipeline and stage metrics
	* Preconditions: none.
	* Postconditions: This method will initialize the pipelineExecMetric to
	*                 default values and remove any prior stageExecMetrics.
	*/
	void initialize();

private:
	// private vars for holding collection of execution stage metrics and
	// total pipeline execution metrics.
	std::vector<StageExecMetrics> stagesExecMetrics;
	PipelineExecMetrics pipelineExecMetrics;
};
