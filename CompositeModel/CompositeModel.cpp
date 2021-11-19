#include <iomanip>
#include <iostream>
#include <fstream>
#include "CompositeModel.h"


void CompositeModel::initialize(SimParameters params, Eigen::VectorXd restPos, std::vector<std::shared_ptr<FiniteElement>> elements, Eigen::VectorXd massVec, std::map<int, double>* clampedPoints)
{
	elements_ = elements;
	restPos_ = restPos;
	params_ = params;

	updateProjM(clampedPoints);
	// assembleMass(massVec);
	massVec_ = massVec;
}

void CompositeModel::updateProjM(std::map<int, double> *clampedPoints)
{
	int row = 0;
	int nverts = restPos_.size();

	fixedPos_.resize(nverts, false);
	std::vector<Eigen::Triplet<double>> T;

	if (clampedPoints)
		clampedPoints_ = *clampedPoints;

	for (int i = 0; i < nverts; i++)
	{
	    if(clampedPoints)
	    {
	        if (clampedPoints->find(i) != clampedPoints->end())
			{
				fixedPos_[i] = true;
				continue;
			}
	            
		}
		T.push_back(Eigen::Triplet<double>(row, i, 1.0));
		row++;
	}
	projM_.resize(row, nverts);
	projM_.setFromTriplets(T.begin(), T.end());

	unProjM_ = projM_.transpose();
}

Eigen::VectorXd CompositeModel::assembleMass(Eigen::VectorXd massVec)
{
	return projM_ * massVec_;
}

void CompositeModel::convertVar2Pos(Eigen::VectorXd q, Eigen::VectorXd &pos)
{
    pos = unProjM_ * q;

    for(auto &it : clampedPoints_)
        pos(it.first) = it.second;
}

void CompositeModel::convertPos2Var(Eigen::VectorXd pos, Eigen::VectorXd &q)
{
    q = projM_ * pos;
}

void CompositeModel::computeCompression(Eigen::VectorXd pos, Eigen::VectorXd& compression)
{
	int nfaces = elements_.size();
	compression.resize(nfaces);
	compression.setZero();
	
	for (int i = 0; i < nfaces; i++)
	{
		int v0 = elements_[i]->vid0_;
		int v1 = elements_[i]->vid1_;

		double drRest = elements_[i]->restP1_ - elements_[i]->restP0_;

		double dr = pos(v1) - pos(v0);
		if (dr / drRest < 1)	// compression does happen
		{
			compression(i) = 1 - dr / drRest;
		}
	}
	
}

double CompositeModel::computeEnergy(Eigen::VectorXd q)
{
	double energy = 0;
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);

	if (params_.gravityEnabled)
		energy += computeGravityPotential(pos);
	if (params_.elasticEnabled)
		energy += computeElasticPotential(pos);
	if (params_.floorEnabled)
		energy += computeFloorBarrier(pos);
	if (params_.internalContactEnabled)
		energy += computeInternalBarrier(pos);
	return energy;
}


// gradient
void CompositeModel::computeGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);
	grad.setZero(pos.rows());
	if (params_.gravityEnabled)
	{
		Eigen::VectorXd gravityGrad;
		computeGravityGradient(pos, gravityGrad);
		grad += gravityGrad;
	}
	if (params_.elasticEnabled)
	{
		Eigen::VectorXd springGrad;
		computeElasticGradient(pos, springGrad);
		grad += springGrad;
	}
	if (params_.floorEnabled)
	{
		Eigen::VectorXd floorGrad;
		computeFloorGradeint(pos, floorGrad);
		grad += params_.barrierStiffness * floorGrad;
	}
	if (params_.internalContactEnabled)
	{
		Eigen::VectorXd internalContactGrad;
		computeInternalGradient(pos, internalContactGrad);
		grad += params_.barrierStiffness * internalContactGrad;
	}
	grad = projM_ * grad;
}

// hessian
void CompositeModel::computeHessian(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian)
{
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);
	//Hessian of gravity is zero
	std::vector<Eigen::Triplet<double>> hessianT;
	if (params_.elasticEnabled)
		computeElasticHessian(pos, hessianT);
	if (params_.floorEnabled)
	{
		std::vector<Eigen::Triplet<double>> floorT;
		computeFloorHessian(pos, floorT);
		for (int i = 0; i < floorT.size(); i++)
		{
			hessianT.push_back(Eigen::Triplet<double>(floorT[i].row(), floorT[i].col(), params_.barrierStiffness * floorT[i].value()));
		}
	}
	if (params_.internalContactEnabled)
	{
		std::vector<Eigen::Triplet<double>> internlT;
		computeInternalHessian(pos, internlT);
		for (int i = 0; i < internlT.size(); i++)
		{
			hessianT.push_back(Eigen::Triplet<double>(internlT[i].row(), internlT[i].col(), params_.barrierStiffness * internlT[i].value()));
		}
	}
	hessian.resize(pos.size(), pos.size());
	hessian.setFromTriplets(hessianT.begin(), hessianT.end());
	hessian = projM_ * hessian * unProjM_;
}



double CompositeModel::computeEnergyPart1(Eigen::VectorXd q)
{
	double energy = 0;
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);

	if (params_.gravityEnabled)
		energy += computeGravityPotential(pos);
	if (params_.elasticEnabled)
		energy += computeElasticPotential(pos);
	return energy;
}


// gradient
void CompositeModel::computeGradientPart1(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);

	grad = Eigen::VectorXd::Zero(pos.size());
	if (params_.gravityEnabled)
	{
		Eigen::VectorXd gravityGrad;
		computeGravityGradient(pos, gravityGrad);
		grad += gravityGrad;
	}
	if (params_.elasticEnabled)
	{
		Eigen::VectorXd springGrad;
		computeElasticGradient(pos, springGrad);
		grad += springGrad;
	}

	grad = projM_ * grad;
	
}

// hessian
void CompositeModel::computeHessianPart1(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian)
{
	//Hessian of gravity is zero
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);

	std::vector<Eigen::Triplet<double>> hessianT;
	if (params_.elasticEnabled)
		computeElasticHessian(pos, hessianT);
	hessian.resize(pos.size(), pos.size());
	hessian.setFromTriplets(hessianT.begin(), hessianT.end());
	hessian = projM_ * hessian * unProjM_;
}


double CompositeModel::computeEnergyPart2(Eigen::VectorXd q)
{
	double energy = 0;
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);
	if (params_.floorEnabled)
		energy += computeFloorBarrier(pos);
	if (params_.internalContactEnabled)
		energy += computeInternalBarrier(pos);
	return energy;
}


// gradient
void CompositeModel::computeGradientPart2(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);
	grad = Eigen::VectorXd::Zero(pos.size());
	if (params_.floorEnabled)
	{
		Eigen::VectorXd floorGrad;
		computeFloorGradeint(pos, floorGrad);
		grad += params_.barrierStiffness * floorGrad;
	}
	if (params_.internalContactEnabled)
	{
		Eigen::VectorXd internalContactGrad;
		computeInternalGradient(pos, internalContactGrad);
		grad += params_.barrierStiffness * internalContactGrad;
	}

	grad = projM_ * grad;
}

// hessian
void CompositeModel::computeHessianPart2(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian)
{
	//Hessian of gravity is zero
	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);
	std::vector<Eigen::Triplet<double>> hessianT;
	if (params_.floorEnabled)
	{
		std::vector<Eigen::Triplet<double>> floorT;
		computeFloorHessian(pos, floorT);
		for (int i = 0; i < floorT.size(); i++)
		{
			hessianT.push_back(Eigen::Triplet<double>(floorT[i].row(), floorT[i].col(), params_.barrierStiffness * floorT[i].value()));
		}
	}
	if (params_.internalContactEnabled)
	{
		std::vector<Eigen::Triplet<double>> internlT;
		computeInternalHessian(pos, internlT);
		for (int i = 0; i < internlT.size(); i++)
		{
			hessianT.push_back(Eigen::Triplet<double>(internlT[i].row(), internlT[i].col(), params_.barrierStiffness * internlT[i].value()));
		}
	}
	hessian.resize(pos.size(), pos.size());
	hessian.setFromTriplets(hessianT.begin(), hessianT.end());

	hessian = projM_ * hessian * unProjM_;
}


// Elastic potential
double CompositeModel::computeElasticPotential(Eigen::VectorXd pos)
{
	double energy = 0;
	for(auto &el : elements_)
	{
		int v0 = el->vid0_;
		int v1 = el->vid1_;

		energy += el->computeElementPotential(pos(v0), pos(v1));
		// break;

	}
	return energy;

}

void CompositeModel::computeElasticGradient(Eigen::VectorXd pos, Eigen::VectorXd& grad)
{
	grad.setZero(pos.rows());

	for(auto &el : elements_)
	{
		int v0 = el->vid0_;
		int v1 = el->vid1_;
		
		Eigen::Vector2d localGrad;

		double localEnergy = el->computeElementPotential(pos(v0), pos(v1), &localGrad);
		grad(v0) += localGrad(0);
		grad(v1) += localGrad(1);
		// break;

	}
}

void CompositeModel::computeElasticHessian(Eigen::VectorXd pos, std::vector<Eigen::Triplet<double>>& T)
{
	for (auto &el : elements_)
	{
		int v0 = el->vid0_;
		int v1 = el->vid1_;
		
		Eigen::Matrix2d localH;

		double localEnergy = el->computeElementPotential(pos(v0), pos(v1), NULL, &localH);

		T.push_back(Eigen::Triplet<double>(v0, v0, localH(0, 0)));
		T.push_back(Eigen::Triplet<double>(v0, v1, localH(0, 1)));
		T.push_back(Eigen::Triplet<double>(v1, v0, localH(1, 0)));
		T.push_back(Eigen::Triplet<double>(v1, v1, localH(1, 1)));
		// break;
	}
}

// floor barrier function
double CompositeModel::computeFloorBarrier(Eigen::VectorXd pos)
{
	double barrier = 0.0;
	int nverts = restPos_.size();
	for (int i = 0; i < nverts; i++)
	{
		if (fixedPos_[i])
			continue;
		
		if (pos(i) <= params_.barrierEps)
		{
			double dist = pos(i);
			if (dist <= 1e-10)
				std::cout << "dist to the floor is too small: " << dist << std::endl;
			barrier += -(dist - params_.barrierEps) * (dist - params_.barrierEps) * std::log(dist / params_.barrierEps);
		}
	}
	return barrier;
}

void CompositeModel::computeFloorGradeint(Eigen::VectorXd pos, Eigen::VectorXd& grad)
{
	int nverts = restPos_.size();
	grad.setZero(pos.size());
	for (int i = 0; i < nverts; i++)
	{
		if (fixedPos_[i])
			continue;
		
		if (pos(i) <= params_.barrierEps)
		{
			double dist = pos(i);
			grad(i) += -(dist - params_.barrierEps) * (2 * std::log(dist / params_.barrierEps) - params_.barrierEps / dist + 1);
		}
	}
}

void CompositeModel::computeFloorHessian(Eigen::VectorXd pos, std::vector<Eigen::Triplet<double>>& hessian)
{
	int nverts = restPos_.size();

	for (int i = 0; i < nverts; i++)
	{
		if (fixedPos_[i])
			continue;
		if (pos(i) <= params_.barrierEps)
		{
			double dist = pos(i);
			double value = -2 * std::log(dist / params_.barrierEps) + (params_.barrierEps - dist) * (params_.barrierEps + 3 * dist) / (dist * dist);
			hessian.push_back(Eigen::Triplet<double>(i, i, value));
		}
	}
}

// gravity
double CompositeModel::computeGravityPotential(Eigen::VectorXd pos)
{
	int nverts = restPos_.size();
	double gpotential = 0.0;

	for (size_t i = 0; i < nverts; i++)
	{
		if (fixedPos_[i])
			continue;
		gpotential -= params_.gravityG * massVec_(i) * pos(i);
	}

	return gpotential;
}

void CompositeModel::computeGravityGradient(Eigen::VectorXd pos, Eigen::VectorXd& grad)
{

	int nverts = restPos_.size();
	grad.setZero(pos.size());

	for (size_t i = 0; i < nverts; i++)
	{
		if (fixedPos_[i])
			continue;
		grad(i) = -params_.gravityG * massVec_(i);
	}
}

// internal contact barrier
double CompositeModel::computeInternalBarrier(Eigen::VectorXd pos)
{
	double energy = 0;
	for (auto &el : elements_)
	{
		int v0 = el->vid0_;
		int v1 = el->vid1_;
		if(fixedPos_[v0] && fixedPos_[v1])
			continue;

		double dr = pos(v1) - pos(v0);

		double dist = std::abs(dr);
		if (dist < params_.barrierEps)
		{
		    if(dist <= 1e-10)
		        std::cout << "dist of two points are too small: " << dist << std::endl;
			energy += -(dist - params_.barrierEps) * (dist - params_.barrierEps) * std::log(dist / params_.barrierEps);
		}
	}
	return energy;
}

void CompositeModel::computeInternalGradient(Eigen::VectorXd pos, Eigen::VectorXd& grad)
{
	grad.setZero(pos.size());

	for (auto &el : elements_)
	{
		int v0 = el->vid0_;
		int v1 = el->vid1_;

		if(fixedPos_[v0] && fixedPos_[v1])
			continue;

		double dr = pos(v1) - pos(v0);

		double dist = std::abs(dr);
		if (dist < params_.barrierEps)
		{
			double tempValue = -(dist - params_.barrierEps) * (2 * std::log(dist / params_.barrierEps) - params_.barrierEps / dist + 1);
			
			if (dr < 0)
			{
				grad(v0) += tempValue;
				grad(v1) += -tempValue;
			}
			else
			{
				grad(v0) += -tempValue;
				grad(v1) += tempValue;
			}
		}
	}
}

void CompositeModel::computeInternalHessian(Eigen::VectorXd pos, std::vector<Eigen::Triplet<double>>& hessian)
{
	for (auto &el : elements_)
	{
		int v0 = el->vid0_;
		int v1 = el->vid1_;

		if(fixedPos_[v0] && fixedPos_[v1])
			continue;

		double dr = pos(v1) - pos(v0);

		double dist = std::abs(dr);
		if (dist < params_.barrierEps)
		{
			double tempValue = -2 * std::log(dist / params_.barrierEps) + (params_.barrierEps - dist) * (params_.barrierEps + 3 * dist) / (dist * dist);
			
			hessian.push_back(Eigen::Triplet<double>(v0, v0, tempValue));
			hessian.push_back(Eigen::Triplet<double>(v0, v1, -tempValue));
		    hessian.push_back(Eigen::Triplet<double>(v1, v0, -tempValue));
		    hessian.push_back(Eigen::Triplet<double>(v1, v1, tempValue));

		}
	}
}

double CompositeModel::getMaxStepSize(Eigen::VectorXd q, Eigen::VectorXd dir)
{
	double maxStep = 1.0;

	Eigen::VectorXd pos;
	convertVar2Pos(q, pos);
	dir = unProjM_ * dir;

	int nverts = pos.size();
	if (params_.materialType == SimParameters::MT_NEOHOOKEAN)
	{
		for(auto &el : elements_)
		{
			int v0 = el->vid0_;
			int v1 = el->vid1_;

			double dr = pos(v1) - pos(v0);
			double deltaDr = dir(v1) - dir(v0);

			if (dr * (dr + deltaDr) <= 0) // possible inverse
			{
				double step = -dr / deltaDr;
				if (step > 0)
					maxStep = std::min(maxStep, 0.8 * step);
			}
		}
	}

	if (params_.floorEnabled)
	{
		for (int i = 0; i < nverts; i++)
		{
			
			if (fixedPos_[i])
				continue;

			double upperStep = 1.0;
			double lowerStep = (-pos(i)) / dir(i) > 0 ? (-pos(i)) / dir(i) : 1.0;
			if(upperStep > 0 || lowerStep > 0)
			{
			    double qiStep = 0.8 * std::min(upperStep, lowerStep);
			    maxStep = std::min(maxStep, qiStep);
			}
		}
	}
//    std::cout << "max step size: " << maxStep << std::endl;
	return maxStep;
}

void CompositeModel::testPotentialDifferential(Eigen::VectorXd q)
{
	Eigen::VectorXd direction = Eigen::VectorXd::Random(q.size());
	direction.normalize();

	double V = computeEnergy(q);
	Eigen::VectorXd g;
	computeGradient(q, g);

	for (int k = 4; k <= 12; k++)
	{
		double eps = pow(10, -k);
		double  epsV = computeEnergy(q + eps * direction);
		std::cout << "Epsilon = " << eps << std::endl;
		std::cout << "Finite difference: " << (epsV - V) / eps << std::endl;
		std::cout << "directional derivative: " << g.dot(direction) << std::endl;
		std::cout << "The difference between above two is: " << abs((epsV - V) / eps - g.dot(direction)) << std::endl << std::endl;
	}
}

void CompositeModel::testGradientDifferential(Eigen::VectorXd q)
{
	Eigen::VectorXd direction = Eigen::VectorXd::Random(q.size());
	direction.normalize();
	Eigen::VectorXd g;
	computeGradient(q, g);

	Eigen::SparseMatrix<double> H(q.size(), q.size());


	computeHessian(q, H);

	for (int k = 1; k <= 12; k++)
	{
		double eps = pow(10, -k);

		Eigen::VectorXd epsF;
		computeGradient(q + eps * direction, epsF);

		std::cout << "EPS is: " << eps << std::endl;
		std::cout << "Norm of Finite Difference is: " << (epsF - g).norm() / eps << std::endl;
		std::cout << "Norm of Directinal Gradient is: " << (H * direction).norm() << std::endl;
		std::cout << "The difference between above two is: " << ((epsF - g) / eps - H * direction).norm() << std::endl << std::endl;

	}
}

// void CompositeModel::testPotentialDifferentialPerface(Eigen::VectorXd q, int faceId)
// {
//     Eigen::Vector2d direction = Eigen::Vector2d::Random();

//     int v0 = restF_(faceId, 0);
//     int v1 = restF_(faceId, 1);

//     int reducedV0 = indexMap_[v0];
//     int reducedV1 = indexMap_[v1];

//     double V = computeElasticPotentialPerface(q, faceId);
//     Eigen::Vector2d g;
//     computeElasticGradientPerface(q, faceId, g);

//     for (int k = 4; k <= 12; k++)
//     {
//         double eps = pow(10, -k);
//         Eigen::VectorXd epsQ = q;
//         if(reducedV0 != -1)
//             epsQ(reducedV0) = q(reducedV0) + eps * direction(0);
//         if(reducedV1 != -1)
//             epsQ(reducedV1) = q(reducedV1) + eps * direction(1);

//         double  epsV = computeElasticPotentialPerface(epsQ, faceId);
//         std::cout << "Epsilon = " << eps << std::endl;
//         std::cout << "Finite difference: " << (epsV - V) / eps << std::endl;
//         std::cout << "directional derivative: " << g.dot(direction) << std::endl;
//         std::cout << "The difference between above two is: " << abs((epsV - V) / eps - g.dot(direction)) << std::endl << std::endl;
//     }
// }

// void CompositeModel::testGradientDifferentialPerface(Eigen::VectorXd q, int faceId)
// {
//     Eigen::Vector2d direction = Eigen::Vector2d::Random();
//     direction.normalize();
//     Eigen::Vector2d g;
//     computeElasticGradientPerface(q, faceId, g);

//     Eigen::Matrix2d H;
//     int v0 = restF_(faceId, 0);
//     int v1 = restF_(faceId, 1);

//     int reducedV0 = indexMap_[v0];
//     int reducedV1 = indexMap_[v1];

//     computeElasticHessianPerface(q, faceId, H);

//     for (int k = 1; k <= 12; k++)
//     {
//         double eps = pow(10, -k);
//         Eigen::VectorXd epsQ = q;
//         if(reducedV0 != -1)
//             epsQ(reducedV0) = q(reducedV0) + eps * direction(0);
//         if(reducedV1 != -1)
//             epsQ(reducedV1) = q(reducedV1) + eps * direction(1);

//         Eigen::Vector2d epsF;
//         computeElasticGradientPerface(epsQ, faceId, epsF);

//         std::cout << "EPS is: " << eps << std::endl;
//         std::cout << "Norm of Finite Difference is: " << (epsF - g).norm() / eps << std::endl;
//         std::cout << "Norm of Directinal Gradient is: " << (H * direction).norm() << std::endl;
//         std::cout << "The difference between above two is: " << ((epsF - g) / eps - H * direction).norm() << std::endl << std::endl;

//     }
// }

// void CompositeModel::testFloorBarrierEnergy(Eigen::VectorXd q)
// {
//     Eigen::VectorXd testQ = q;

//     // first make sure that testQ fall into the validation domain of the IPC barrier
//     for(int i = 0; i < testQ.size(); i++)
//     {
//         testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
//     }

//     Eigen::VectorXd grad;
//     double E = computeFloorBarrier(testQ);
//     computeFloorGradeint(testQ, grad);

//     Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

//     for(int i = 3; i <= 9; i++)
//     {
//         double eps = std::pow(0.1, i);
//         Eigen::VectorXd newQ = testQ + eps * dir;

//         double E1 = computeFloorBarrier(newQ);
//         std::cout << "eps: " << eps << std::endl;
//         std::cout << "finite difference: " << (E1 - E) / eps << ", directional derivative: " << dir.dot(grad) << ", error: " << std::abs((E1 - E) / eps - dir.dot(grad)) << std::endl;
//     }
// }

// void CompositeModel::testFloorBarrierGradient(Eigen::VectorXd q)
// {
//     Eigen::VectorXd testQ = q;

//     // first make sure that testQ fall into the validation domain of the IPC barrier
//     for(int i = 0; i < testQ.size(); i++)
//     {
//         testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
//     }

//     Eigen::VectorXd grad;
//     Eigen::SparseMatrix<double> H;
//     std::vector<Eigen::Triplet<double>> T;

//     computeFloorGradeint(testQ, grad);
//     computeFloorHessian(testQ, T);
//     H.resize(q.size(), q.size());
//     H.setFromTriplets(T.begin(), T.end());

//     Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

//     for(int i = 3; i <= 9; i++)
//     {
//         double eps = std::pow(0.1, i);
//         Eigen::VectorXd newQ = testQ + eps * dir;

//         Eigen::VectorXd grad1;
//         computeFloorGradeint(newQ, grad1);
//         std::cout << "eps: " << eps << std::endl;
//         std::cout << "finite difference: " << ((grad1 - grad) / eps).norm() << ", directional derivative: " << (H * dir).norm() << ", error: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
//     }
// }

// void CompositeModel::testInternalBarrierEnergy(Eigen::VectorXd q)
// {
//     int vid0 = restF_(0, 0);
//     int vid1 = restF_(0, 1);

//     Eigen::VectorXd testQ = q;
//     testQ(vid0) = testQ(vid1) + params_.barrierEps * 0.5;

//     Eigen::VectorXd grad;
//     double E = computeInternalBarrier(testQ);
//     computeInternalGradient(testQ, grad);

//     Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

//     for(int i = 3; i <= 9; i++)
//     {
//         double eps = std::pow(0.1, i);
//         Eigen::VectorXd newQ = testQ + eps * dir;

//         double E1 = computeInternalBarrier(newQ);
//         std::cout << "eps: " << eps << std::endl;
//         std::cout << "finite difference: " << (E1 - E) / eps << ", directional derivative: " << dir.dot(grad) << ", error: " << std::abs((E1 - E) / eps - dir.dot(grad)) << std::endl;
//     }


// }

// void CompositeModel::testInternalBarrierGradient(Eigen::VectorXd q)
// {
//     int vid0 = restF_(0, 0);
//     int vid1 = restF_(0, 1);

//     Eigen::VectorXd testQ = q;
//     testQ(vid0) = testQ(vid1) + params_.barrierEps * 0.5;

//     Eigen::VectorXd grad;
//     Eigen::SparseMatrix<double> H;
//     std::vector<Eigen::Triplet<double>> T;

//     computeInternalGradient(testQ, grad);
//     computeInternalHessian(testQ, T);
//     H.resize(q.size(), q.size());
//     H.setFromTriplets(T.begin(), T.end());

//     Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

//     for(int i = 3; i <= 9; i++)
//     {
//         double eps = std::pow(0.1, i);
//         Eigen::VectorXd newQ = testQ + eps * dir;

//         Eigen::VectorXd grad1;
//         computeInternalGradient(newQ, grad1);
//         std::cout << "eps: " << eps << std::endl;
//         std::cout << "finite difference: " << ((grad1 - grad) / eps).norm() << ", directional derivative: " << (H * dir).norm() << ", error: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
//     }
// }

// void CompositeModel::testElasticEnergy(Eigen::VectorXd q)
// {
// 	Eigen::VectorXd testQ = q;

// 	// first make sure that testQ fall into the validation domain of the IPC barrier
// 	for (int i = 0; i < testQ.size(); i++)
// 	{
// 		testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
// 	}

// 	Eigen::VectorXd grad;
// 	double E = computeElasticPotential(testQ);
// 	computeElasticGradient(testQ, grad);

// 	Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

// 	for (int i = 3; i <= 9; i++)
// 	{
// 		double eps = std::pow(0.1, i);
// 		Eigen::VectorXd newQ = testQ + eps * dir;

// 		double E1 = computeElasticPotential(newQ);
// 		std::cout << "eps: " << eps << std::endl;
// 		std::cout << "finite difference: " << (E1 - E) / eps << ", directional derivative: " << dir.dot(grad) << ", error: " << std::abs((E1 - E) / eps - dir.dot(grad)) << std::endl;
// 	}
// }

// void CompositeModel::testElasticGradient(Eigen::VectorXd q)
// {
// 	Eigen::VectorXd testQ = q;

// 	// first make sure that testQ fall into the validation domain of the IPC barrier
// 	for (int i = 0; i < testQ.size(); i++)
// 	{
// 		testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
// 	}

// 	Eigen::VectorXd grad;
// 	Eigen::SparseMatrix<double> H;
// 	std::vector<Eigen::Triplet<double>> T;

// 	computeElasticGradient(testQ, grad);
// 	computeElasticHessian(testQ, T);
// 	H.resize(q.size(), q.size());
// 	H.setFromTriplets(T.begin(), T.end());

// 	Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

// 	for (int i = 3; i <= 9; i++)
// 	{
// 		double eps = std::pow(0.1, i);
// 		Eigen::VectorXd newQ = testQ + eps * dir;

// 		Eigen::VectorXd grad1;
// 		computeElasticGradient(newQ, grad1);
// 		std::cout << "eps: " << eps << std::endl;
// 		std::cout << "finite difference: " << ((grad1 - grad) / eps).norm() << ", directional derivative: " << (H * dir).norm() << ", error: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
// 	}
// }