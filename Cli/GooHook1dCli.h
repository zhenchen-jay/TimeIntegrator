#pragma once

#include <iostream>
#include <deque>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>

#include "../PhysicalModel/GooHook1d.h"
#include "../PhysicalModel/SimParameters.h"
#include "../PhysicalModel/SceneObjects.h"

// We fixed the x coordinate 

class GooHook1dCli
{
public:
	GooHook1dCli()
	{ 
		time_ = 0;
		iterNum_ = 0;
	}

	void initSimulation(std::string outputFolder = "../output/");

	bool simulateOneStep();

	void updateParams()
	{
		model_.params_ = params_;
		getOutputFolderPath();
	}

	void getOutputFolderPath();

	bool reachTheTermination()
	{
		return time_ >= totalTime_;
	}

	void saveInfo();
	
public:
	SimParameters params_;

	GooHook1d model_;
	double totalTime_;
	int totalIterNum_;

	std::string outputFolderPath_, baseFolder_;
	double time_;
	int iterNum_;
};



