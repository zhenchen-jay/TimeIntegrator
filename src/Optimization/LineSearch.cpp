
#include "../../include/Optimization/LineSearch.h"
#include <iostream>

double LineSearch::backtrackingArmijo(const Eigen::VectorXd& x, const Eigen::VectorXd& grad, const Eigen::VectorXd& dir, std::function<double(Eigen::VectorXd, Eigen::VectorXd*, Eigen::SparseMatrix<double>*)> objFunc, const double alphaInit)
{
    const double c = 0.2;
    const double rho = 0.5;
    double alpha = alphaInit;

    Eigen::VectorXd xNew = x + alpha * dir;
    double fNew = objFunc(xNew, NULL, NULL);
    double f = objFunc(x, NULL, NULL);
    const double cache = c * grad.dot(dir);

    while (fNew > f + alpha * cache) {
        alpha *= rho;
        xNew = x + alpha * dir;
        fNew = objFunc(xNew, NULL, NULL);
        /*std::cout << "cache: " << cache << ", alpha: " << alpha << ", f: " << f << ", fNew: " << fNew << std::endl;
        std::cout << "xnew: " << xNew.transpose() << std::endl;*/
    }

    return alpha;
}
