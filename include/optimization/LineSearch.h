#pragma once
#include <Eigen/Dense>

template <typename Problem>
double backtrackingArmijo(const Eigen::VectorXd& x, const Eigen::VectorXd& grad, const Eigen::VectorXd& dir, Problem& objFunc, const double alphaInit = 1.0)
{
    const double c = 0.2;
    const double rho = 0.5;
    double alpha = alphaInit;

    Eigen::VectorXd xNew = x + alpha * dir;
    double fNew = objFunc.value(xNew);
    double f = objFunc.value(x);
    const double cache = c * grad.dot(dir);

    while (fNew > f + alpha * cache) {
        alpha *= rho;
        xNew = x + alpha * dir;
        fNew = objFunc.value(xNew);
    }

    return alpha;
}