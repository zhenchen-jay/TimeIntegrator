#include <iomanip>
#include <iostream>
#include <fstream>
#include "PhysicalModel.h"


void PhysicalModel::initialize(Eigen::VectorXd restPos, Eigen::MatrixXi restF, Eigen::VectorXd massVec, std::map<int, double>* clampedPoints)
{
	restPos_ = restPos;
	curPos_ = restPos;
	restF_ = restF;
	massVec_ = massVec;
	updateProjM(clampedPoints);
	assembleMass(massVec);
}

void PhysicalModel::updateProjM(std::map<int, double> *clampedPoints)
{
	int row = 0;
	int nverts = restPos_.size();
	clampedPos_.clear();
	if(clampedPoints)
	    clampedPos_ = *clampedPoints;

	std::vector<Eigen::Triplet<double>> T;
	indexMap_.resize(nverts, -1);

	for (int i = 0; i < nverts; i++)
	{
	    if(clampedPoints)
	    {
	        if (clampedPoints->find(i) != clampedPoints->end())
	            continue;
	    }


		T.push_back(Eigen::Triplet<double>(row, i, 1.0));
		indexMap_[i] = row;
		indexInvMap_.push_back(i);
		row++;
	}
	projM_.resize(row, nverts);
	projM_.setFromTriplets(T.begin(), T.end());

	unProjM_ = projM_.transpose();
}

void PhysicalModel::assembleMass(Eigen::VectorXd massVec)
{
	int nverts = restPos_.rows();
	int nVals = projM_.rows();

	massVec_.setZero(nVals);

	std::vector<Eigen::Triplet<double>> coef;

	for (size_t i = 0; i < nverts; i++)
	{
		if (indexMap_[i] == -1)
			continue;
		massVec_(indexMap_[i]) = massVec(i);
	}
}

void PhysicalModel::convertVar2Pos(Eigen::VectorXd q, Eigen::VectorXd &pos)
{
    int nVals = q.size();
    int nverts = restPos_.rows();
    pos.setZero(nverts);

    for(int i = 0; i < nVals; i++)
        pos(indexInvMap_[i]) = q(i);

    for(auto &it : clampedPos_)
        pos(it.first) = it.second;
}

void PhysicalModel::convertPos2Var(Eigen::VectorXd pos, Eigen::VectorXd &q)
{
    int nverts = curPos_.rows();
    int nVals = indexInvMap_.size();

    q.setZero(nVals);

    for(int i = 0; i < nverts; i++)
    {
        if(indexMap_[i] != -1)
            q(indexMap_[i]) = pos(i);
    }
}

double PhysicalModel::computeEnergy(Eigen::VectorXd q)
{
	double energy = 0;

	if (params_.gravityEnabled)
		energy += computeGravityPotential(q);
	if (params_.elasticEnabled)
		energy += computeElasticPotential(q);
	if (params_.floorEnabled)
		energy += computeFloorBarrier(q);
	if (params_.internalContactEnabled)
		energy += computeInternalBarrier(q);
	return energy;
}


// gradient
void PhysicalModel::computeGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	grad = Eigen::VectorXd::Zero(q.size());
	if (params_.gravityEnabled)
	{
		Eigen::VectorXd gravityGrad;
		computeGravityGradient(q, gravityGrad);
		grad += gravityGrad;
	}
	if (params_.elasticEnabled)
	{
		Eigen::VectorXd springGrad;
		computeElasticGradient(q, springGrad);
		grad += springGrad;
	}
	if (params_.floorEnabled)
	{
		Eigen::VectorXd floorGrad;
		computeFloorGradeint(q, floorGrad);
		grad += params_.barrierStiffness * floorGrad;
	}
	if (params_.internalContactEnabled)
	{
		Eigen::VectorXd internalContactGrad;
		computeInternalGradient(q, internalContactGrad);
		grad += params_.barrierStiffness * internalContactGrad;
	}
}

// hessian
void PhysicalModel::computeHessian(Eigen::VectorXd q, Eigen::SparseMatrix<double>& hessian)
{
	//Hessian of gravity is zero
	std::vector<Eigen::Triplet<double>> hessianT;
	if (params_.elasticEnabled)
		computeElasticHessian(q, hessianT);
	if (params_.floorEnabled)
	{
		std::vector<Eigen::Triplet<double>> floorT;
		computeFloorHessian(q, floorT);
		for (int i = 0; i < floorT.size(); i++)
		{
			hessianT.push_back(Eigen::Triplet<double>(floorT[i].row(), floorT[i].col(), params_.barrierStiffness * floorT[i].value()));
		}
	}
	if (params_.internalContactEnabled)
	{
		std::vector<Eigen::Triplet<double>> internlT;
		computeInternalHessian(q, internlT);
		for (int i = 0; i < internlT.size(); i++)
		{
			hessianT.push_back(Eigen::Triplet<double>(internlT[i].row(), internlT[i].col(), params_.barrierStiffness * internlT[i].value()));
		}
	}
	hessian.resize(q.size(), q.size());
	hessian.setFromTriplets(hessianT.begin(), hessianT.end());
}


// Elastic potential
double PhysicalModel::computeElasticPotential(Eigen::VectorXd q)
{
	double energy = 0;
	int nfaces = restF_.rows();
	for (int i = 0; i < nfaces; i++)
	{
		energy += computeElasticPotentialPerface(q, i);
	}
	return energy;

}

void PhysicalModel::computeElasticGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	int nReducedVerts = q.size();
	int nfaces = restF_.rows();
	grad.setZero(nReducedVerts);

	for (int i = 0; i < nfaces; i++)
	{
		Eigen::Vector2d localGrad;
		computeElasticGradientPerface(q, i, localGrad);

		int v0 = restF_(i, 0);
		int v1 = restF_(i, 1);

		int reducedV0 = indexMap_[v0];
		int reducedV1 = indexMap_[v1];

		if (reducedV0 == -1)
		{
			if (reducedV1 == -1)
				continue;
			else
				grad(reducedV1) += localGrad(1);
		}
		else
		{
			if (reducedV1 == -1)
				grad(reducedV0) += localGrad(0);

			else
			{
				grad(reducedV0) += localGrad(0);
				grad(reducedV1) += localGrad(1);
			}

		}

	}
}

void PhysicalModel::computeElasticHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& T)
{
	int nReducedVerts = q.size();
	int nfaces = restF_.rows();

	for (int i = 0; i < nfaces; i++)
	{
		Eigen::Matrix2d localH;
		computeElasticHessianPerface(q, i, localH);

		int v0 = restF_(i, 0);
		int v1 = restF_(i, 1);

		int reducedV0 = indexMap_[v0];
		int reducedV1 = indexMap_[v1];

		if (reducedV0 == -1)
		{
			if (reducedV1 == -1)
				continue;
			else
				T.push_back(Eigen::Triplet<double>(reducedV1, reducedV1, localH(1, 1)));
		}
		else
		{
			if (reducedV1 == -1)
				T.push_back(Eigen::Triplet<double>(reducedV0, reducedV0, localH(0, 0)));

			else
			{
				T.push_back(Eigen::Triplet<double>(reducedV0, reducedV0, localH(0, 0)));
				T.push_back(Eigen::Triplet<double>(reducedV0, reducedV1, localH(0, 1)));
				T.push_back(Eigen::Triplet<double>(reducedV1, reducedV0, localH(1, 0)));
				T.push_back(Eigen::Triplet<double>(reducedV1, reducedV1, localH(1, 1)));
			}

		}

	}
}

// floor barrier function
double PhysicalModel::computeFloorBarrier(Eigen::VectorXd q)
{
	double barrier = 0.0;
	int nverts = restPos_.size();
	for (int i = 0; i < nverts; i++)
	{
		if (indexMap_[i] == -1)
			continue;
		/*if (q(indexMap_[i]) <= params_.barrierEps || q(indexMap_[i]) >= params_.topLine - params_.barrierEps)
		{
			double pos = q(indexMap_[i]);
			double dist = params_.barrierEps;
			if (q(indexMap_[i]) <= params_.barrierEps)
				dist = q(indexMap_[i]);
			else
			    dist = params_.topLine - q(indexMap_[i]);
			if(dist <= 1e-10)
			    std::cout << "dist to the floor is too small: " << dist << std::endl;
			barrier += -(dist - params_.barrierEps) * (dist - params_.barrierEps) * std::log(dist / params_.barrierEps);
		}*/

		if (q(indexMap_[i]) <= params_.barrierEps)
		{
			double pos = q(indexMap_[i]);
			double dist = params_.barrierEps;
			dist = q(indexMap_[i]);
			if (dist <= 1e-10)
				std::cout << "dist to the floor is too small: " << dist << std::endl;
			barrier += -(dist - params_.barrierEps) * (dist - params_.barrierEps) * std::log(dist / params_.barrierEps);
		}
	}
	return barrier;
}

void PhysicalModel::computeFloorGradeint(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	int nverts = restPos_.size();
	grad.setZero(q.size());
	for (int i = 0; i < nverts; i++)
	{
		if (indexMap_[i] == -1)
			continue;
		int id = indexMap_[i];
		/*if (q(id) <= params_.barrierEps || q(id) >= params_.topLine - params_.barrierEps)
		{
			double dist = params_.barrierEps;
			if (q(id) <= params_.barrierEps)
				dist = q(id);
			else
			    dist = params_.topLine - q(id);
			grad(id) += -(dist - params_.barrierEps) * (2 * std::log(dist / params_.barrierEps) - params_.barrierEps / dist + 1);
			if (q(id) >= params_.topLine - params_.barrierEps)
				grad(id) *= -1;
		}*/

		if (q(id) <= params_.barrierEps)
		{
			double dist = params_.barrierEps;
			dist = q(id);
			grad(id) += -(dist - params_.barrierEps) * (2 * std::log(dist / params_.barrierEps) - params_.barrierEps / dist + 1);
			if (q(id) >= params_.topLine - params_.barrierEps)
				grad(id) *= -1;
		}
	}
}

void PhysicalModel::computeFloorHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& hessian)
{
	int nverts = restPos_.size();

	for (int i = 0; i < nverts; i++)
	{
		if (indexMap_[i] == -1)
			continue;
		int id = indexMap_[i];
		/*if (q(id) <= params_.barrierEps || q(id) >= params_.topLine - params_.barrierEps)
		{
			double dist = params_.barrierEps;
			if (q(id) <= params_.barrierEps)
				dist = q(id);
			else
			    dist = params_.topLine - q(id);
			double value = -2 * std::log(dist / params_.barrierEps) + (params_.barrierEps - dist) * (params_.barrierEps + 3 * dist) / (dist * dist);
			hessian.push_back(Eigen::Triplet<double>(id, id, value));
		}*/
		if (q(id) <= params_.barrierEps)
		{
			double dist = params_.barrierEps;
			dist = q(id);
			double value = -2 * std::log(dist / params_.barrierEps) + (params_.barrierEps - dist) * (params_.barrierEps + 3 * dist) / (dist * dist);
			hessian.push_back(Eigen::Triplet<double>(id, id, value));
		}
	}
}

// gravity
double PhysicalModel::computeGravityPotential(Eigen::VectorXd q)
{
	int nverts = restPos_.size();
	double gpotential = 0.0;

	for (size_t i = 0; i < nverts; i++)
	{
		if (indexMap_[i] == -1)
			continue;
		gpotential -= params_.gravityG * massVec_(indexMap_[i]) * q(indexMap_[i]);
	}

	return gpotential;
}

void PhysicalModel::computeGravityGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{

	int nverts = restPos_.size();
	grad.setZero(q.size());

	for (size_t i = 0; i < nverts; i++)
	{
		if (indexMap_[i] == -1)
			continue;
		grad(indexMap_[i]) = -params_.gravityG * massVec_(indexMap_[i]);
	}
}

// internal contact barrier
double PhysicalModel::computeInternalBarrier(Eigen::VectorXd q)
{
	double energy = 0;
	int nfaces = restF_.rows();

	for (int i = 0; i < nfaces; i++)
	{
		int v0 = restF_(i, 0);
		int v1 = restF_(i, 1);

		int reducedV0 = indexMap_[v0];
		int reducedV1 = indexMap_[v1];

		double drRest = restPos_(v1) - restPos_(v0);
		double dr = 0;

		if (reducedV0 == -1)
		{
			if (reducedV1 == -1)
				continue;
			dr = q(reducedV1) - restPos_(v0);
		}
		else
		{
			if (reducedV1 == -1)
				dr = restPos_(v1) - q(reducedV0);
			else
				dr = q(reducedV1) - q(reducedV0);

		}

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

void PhysicalModel::computeInternalGradient(Eigen::VectorXd q, Eigen::VectorXd& grad)
{
	grad.setZero(q.size());

	int nfaces = restF_.rows();

	for (int i = 0; i < nfaces; i++)
	{
		int v0 = restF_(i, 0);
		int v1 = restF_(i, 1);

		int reducedV0 = indexMap_[v0];
		int reducedV1 = indexMap_[v1];

		double drRest = restPos_(v1) - restPos_(v0);
		double dr = 0;

		if (reducedV0 == -1)
		{
			dr = q(reducedV1) - restPos_(v0);
		}
		else
		{
			if (reducedV1 == -1)
				dr = restPos_(v1) - q(reducedV0);
			else
				dr = q(reducedV1) - q(reducedV0);

		}

		double dist = std::abs(dr);
		if (dist < params_.barrierEps)
		{
			double tempValue = -(dist - params_.barrierEps) * (2 * std::log(dist / params_.barrierEps) - params_.barrierEps / dist + 1);
			if (reducedV0 == -1)
			{
				if (reducedV1 == -1)
					continue;
				if (dr < 0)
					grad(reducedV1) += -tempValue;
				else
					grad(reducedV1) += tempValue;

			}
			else
			{
				if (reducedV1 == -1)
				{
					if (dr < 0)
						grad(reducedV0) += tempValue;
					else
						grad(reducedV0) += -tempValue;
				}
				else
				{
					if (dr < 0)
					{
						grad(reducedV0) += tempValue;
						grad(reducedV1) += -tempValue;
					}
					else
					{
						grad(reducedV0) += -tempValue;
						grad(reducedV1) += tempValue;
					}
				}
			}
		}
	}
}

void PhysicalModel::computeInternalHessian(Eigen::VectorXd q, std::vector<Eigen::Triplet<double>>& hessian)
{
	int nfaces = restF_.rows();

	for (int i = 0; i < nfaces; i++)
	{
		int v0 = restF_(i, 0);
		int v1 = restF_(i, 1);

		int reducedV0 = indexMap_[v0];
		int reducedV1 = indexMap_[v1];

		double drRest = restPos_(v1) - restPos_(v0);
		double dr = 0;

		if (reducedV0 == -1)
		{
			dr = q(reducedV1) - restPos_(v0);
		}
		else
		{
			if (reducedV1 == -1)
				dr = restPos_(v1) - q(reducedV0);
			else
				dr = q(reducedV1) - q(reducedV0);

		}

		double dist = std::abs(dr);
		if (dist < params_.barrierEps)
		{
			double tempValue = -2 * std::log(dist / params_.barrierEps) + (params_.barrierEps - dist) * (params_.barrierEps + 3 * dist) / (dist * dist);
			if (reducedV0 == -1)
			{
				if (reducedV1 == -1)
					continue;
				hessian.push_back(Eigen::Triplet<double>(reducedV1, reducedV1, tempValue));

			}
			else
			{
				if (reducedV1 == -1)
				{
					hessian.push_back(Eigen::Triplet<double>(reducedV0, reducedV0, tempValue));
				}
				else
				{
				    hessian.push_back(Eigen::Triplet<double>(reducedV0, reducedV0, tempValue));
				    hessian.push_back(Eigen::Triplet<double>(reducedV0, reducedV1, -tempValue));
				    hessian.push_back(Eigen::Triplet<double>(reducedV1, reducedV0, -tempValue));
				    hessian.push_back(Eigen::Triplet<double>(reducedV1, reducedV1, tempValue));
				}
			}
		}
	}
}

double PhysicalModel::getMaxStepSize(Eigen::VectorXd q, Eigen::VectorXd dir)
{
	int nverts = restPos_.size();
	int nfaces = restF_.rows();
	double maxStep = 1.0;

	if (params_.materialType == SimParameters::MT_NEOHOOKEAN)
	{
		for (int i = 0; i < nfaces; i++)
		{
			int v0 = restF_(i, 0);
			int v1 = restF_(i, 1);

			int reducedV0 = indexMap_[v0];
			int reducedV1 = indexMap_[v1];

			double dr = 0, deltaDr = 0;
			if (reducedV0 == -1)
			{
				dr = q(reducedV1) - restPos_(v0);
				deltaDr = dir(reducedV1);
			}
			else
			{
				if (reducedV1 == -1)
				{
					dr = restPos_(v1) - q(reducedV0);
					deltaDr = -dir(reducedV0);
				}

				else
				{
					dr = q(reducedV1) - q(reducedV0);
					deltaDr = dir(reducedV1) - dir(reducedV0);
				}
			}

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
			int id = indexMap_[i];
			if (id == -1)
				continue;

			//double upperStep = (params_.topLine - q(id)) / dir(id) > 0 ? (params_.topLine - q(id)) / dir(id) : 1.0;
			double upperStep = 1.0;
			double lowerStep = (-q(id)) / dir(id) > 0 ? (-q(id)) / dir(id) : 1.0;
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

void PhysicalModel::testPotentialDifferential(Eigen::VectorXd q)
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

void PhysicalModel::testGradientDifferential(Eigen::VectorXd q)
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

void PhysicalModel::testPotentialDifferentialPerface(Eigen::VectorXd q, int faceId)
{
    Eigen::Vector2d direction = Eigen::Vector2d::Random();

    int v0 = restF_(faceId, 0);
    int v1 = restF_(faceId, 1);

    int reducedV0 = indexMap_[v0];
    int reducedV1 = indexMap_[v1];

    double V = computeElasticPotentialPerface(q, faceId);
    Eigen::Vector2d g;
    computeElasticGradientPerface(q, faceId, g);

    for (int k = 4; k <= 12; k++)
    {
        double eps = pow(10, -k);
        Eigen::VectorXd epsQ = q;
        if(reducedV0 != -1)
            epsQ(reducedV0) = q(reducedV0) + eps * direction(0);
        if(reducedV1 != -1)
            epsQ(reducedV1) = q(reducedV1) + eps * direction(1);

        double  epsV = computeElasticPotentialPerface(epsQ, faceId);
        std::cout << "Epsilon = " << eps << std::endl;
        std::cout << "Finite difference: " << (epsV - V) / eps << std::endl;
        std::cout << "directional derivative: " << g.dot(direction) << std::endl;
        std::cout << "The difference between above two is: " << abs((epsV - V) / eps - g.dot(direction)) << std::endl << std::endl;
    }
}

void PhysicalModel::testGradientDifferentialPerface(Eigen::VectorXd q, int faceId)
{
    Eigen::Vector2d direction = Eigen::Vector2d::Random();
    direction.normalize();
    Eigen::Vector2d g;
    computeElasticGradientPerface(q, faceId, g);

    Eigen::Matrix2d H;
    int v0 = restF_(faceId, 0);
    int v1 = restF_(faceId, 1);

    int reducedV0 = indexMap_[v0];
    int reducedV1 = indexMap_[v1];

    computeElasticHessianPerface(q, faceId, H);

    for (int k = 1; k <= 12; k++)
    {
        double eps = pow(10, -k);
        Eigen::VectorXd epsQ = q;
        if(reducedV0 != -1)
            epsQ(reducedV0) = q(reducedV0) + eps * direction(0);
        if(reducedV1 != -1)
            epsQ(reducedV1) = q(reducedV1) + eps * direction(1);

        Eigen::Vector2d epsF;
        computeElasticGradientPerface(epsQ, faceId, epsF);

        std::cout << "EPS is: " << eps << std::endl;
        std::cout << "Norm of Finite Difference is: " << (epsF - g).norm() / eps << std::endl;
        std::cout << "Norm of Directinal Gradient is: " << (H * direction).norm() << std::endl;
        std::cout << "The difference between above two is: " << ((epsF - g) / eps - H * direction).norm() << std::endl << std::endl;

    }
}

void PhysicalModel::testFloorBarrierEnergy(Eigen::VectorXd q)
{
    Eigen::VectorXd testQ = q;

    // first make sure that testQ fall into the validation domain of the IPC barrier
    for(int i = 0; i < testQ.size(); i++)
    {
        testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
    }

    Eigen::VectorXd grad;
    double E = computeFloorBarrier(testQ);
    computeFloorGradeint(testQ, grad);

    Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

    for(int i = 3; i <= 9; i++)
    {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd newQ = testQ + eps * dir;

        double E1 = computeFloorBarrier(newQ);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "finite difference: " << (E1 - E) / eps << ", directional derivative: " << dir.dot(grad) << ", error: " << std::abs((E1 - E) / eps - dir.dot(grad)) << std::endl;
    }
}

void PhysicalModel::testFloorBarrierGradient(Eigen::VectorXd q)
{
    Eigen::VectorXd testQ = q;

    // first make sure that testQ fall into the validation domain of the IPC barrier
    for(int i = 0; i < testQ.size(); i++)
    {
        testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
    }

    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> H;
    std::vector<Eigen::Triplet<double>> T;

    computeFloorGradeint(testQ, grad);
    computeFloorHessian(testQ, T);
    H.resize(q.size(), q.size());
    H.setFromTriplets(T.begin(), T.end());

    Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

    for(int i = 3; i <= 9; i++)
    {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd newQ = testQ + eps * dir;

        Eigen::VectorXd grad1;
        computeFloorGradeint(newQ, grad1);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "finite difference: " << ((grad1 - grad) / eps).norm() << ", directional derivative: " << (H * dir).norm() << ", error: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
    }
}

void PhysicalModel::testInternalBarrierEnergy(Eigen::VectorXd q)
{
    int vid0 = restF_(0, 0);
    int vid1 = restF_(0, 1);

    Eigen::VectorXd testQ = q;
    testQ(vid0) = testQ(vid1) + params_.barrierEps * 0.5;

    Eigen::VectorXd grad;
    double E = computeInternalBarrier(testQ);
    computeInternalGradient(testQ, grad);

    Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

    for(int i = 3; i <= 9; i++)
    {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd newQ = testQ + eps * dir;

        double E1 = computeInternalBarrier(newQ);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "finite difference: " << (E1 - E) / eps << ", directional derivative: " << dir.dot(grad) << ", error: " << std::abs((E1 - E) / eps - dir.dot(grad)) << std::endl;
    }


}

void PhysicalModel::testInternalBarrierGradient(Eigen::VectorXd q)
{
    int vid0 = restF_(0, 0);
    int vid1 = restF_(0, 1);

    Eigen::VectorXd testQ = q;
    testQ(vid0) = testQ(vid1) + params_.barrierEps * 0.5;

    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> H;
    std::vector<Eigen::Triplet<double>> T;

    computeInternalGradient(testQ, grad);
    computeInternalHessian(testQ, T);
    H.resize(q.size(), q.size());
    H.setFromTriplets(T.begin(), T.end());

    Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

    for(int i = 3; i <= 9; i++)
    {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd newQ = testQ + eps * dir;

        Eigen::VectorXd grad1;
        computeInternalGradient(newQ, grad1);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "finite difference: " << ((grad1 - grad) / eps).norm() << ", directional derivative: " << (H * dir).norm() << ", error: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
    }
}

void PhysicalModel::testElasticEnergy(Eigen::VectorXd q)
{
	Eigen::VectorXd testQ = q;

	// first make sure that testQ fall into the validation domain of the IPC barrier
	for (int i = 0; i < testQ.size(); i++)
	{
		testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
	}

	Eigen::VectorXd grad;
	double E = computeElasticPotential(testQ);
	computeElasticGradient(testQ, grad);

	Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

	for (int i = 3; i <= 9; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd newQ = testQ + eps * dir;

		double E1 = computeElasticPotential(newQ);
		std::cout << "eps: " << eps << std::endl;
		std::cout << "finite difference: " << (E1 - E) / eps << ", directional derivative: " << dir.dot(grad) << ", error: " << std::abs((E1 - E) / eps - dir.dot(grad)) << std::endl;
	}
}

void PhysicalModel::testElasticGradient(Eigen::VectorXd q)
{
	Eigen::VectorXd testQ = q;

	// first make sure that testQ fall into the validation domain of the IPC barrier
	for (int i = 0; i < testQ.size(); i++)
	{
		testQ(i) = params_.barrierEps / testQ.size() * (i + 1);
	}

	Eigen::VectorXd grad;
	Eigen::SparseMatrix<double> H;
	std::vector<Eigen::Triplet<double>> T;

	computeElasticGradient(testQ, grad);
	computeElasticHessian(testQ, T);
	H.resize(q.size(), q.size());
	H.setFromTriplets(T.begin(), T.end());

	Eigen::VectorXd  dir = Eigen::VectorXd::Random(grad.size());

	for (int i = 3; i <= 9; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd newQ = testQ + eps * dir;

		Eigen::VectorXd grad1;
		computeElasticGradient(newQ, grad1);
		std::cout << "eps: " << eps << std::endl;
		std::cout << "finite difference: " << ((grad1 - grad) / eps).norm() << ", directional derivative: " << (H * dir).norm() << ", error: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
	}
}