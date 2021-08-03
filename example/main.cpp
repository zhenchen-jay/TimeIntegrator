#include <Eigen/Core>
#include <iostream>
#include "../include/MatrixVecExp.h"

void testMatrixExp()
{
    int num = 500;
    std::vector<Eigen::Triplet<double> > tripletList;
    for(int i=0;i<num;++i)
        for(int j=0;j<num;++j)
        {
            double v_ij=static_cast <double> (rand()) / static_cast <double> (RAND_MAX);                         //generate random number
            if(v_ij < 0.1 || v_ij > 0.8)
            {
                tripletList.push_back({i,j,v_ij});      //if larger than treshold, insert it
            }
        }
    Eigen::SparseMatrix<double> A(num, num);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    
    Eigen::MatrixXd expA = A.toDense().exp();
    
    Eigen::VectorXd v = Eigen::VectorXd::Random(num);
    Eigen::VectorXd w = v;
    double err, hump;
    matrixVecExp(A, v, 1.0, 30, 1e-7, w, &err, &hump);
    
    std::cout << "difference: " << (expA * v - w).norm() << ", solver error: " << err << ", solver hump: " << hump << std::endl;
    std::cout << w.norm() << ", " << (expA * v).norm() << std::endl;
}



int main(int argc, char* argv[])
{
    testMatrixExp();
    return 0;
}
