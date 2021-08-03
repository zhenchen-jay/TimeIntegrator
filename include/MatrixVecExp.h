#pragma once
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>

bool matrixVecExp(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& v, const double t, const int m, const double tol, Eigen::VectorXd& w, double *err = NULL, double *hump = NULL)
/*
 Implementation of the Krylove algorithm (see paper "Expokit: A Software Package for Computing Matrix Exponentials" for details) to compute the w = exp(t A) v. We adapted this script from the matlab source code "https://www.maths.uq.edu.au/expokit/matlab/"
 @paramIn
 A:     a sparse sqaure matrix
 v:     a vector
 t:     a constant, which always refer as the time
 m:     dimension of the Krylove subspace, the large it is, the lower the program. 
 tol:   termination tolerance, which controls the accuracy
 
 @paramOut
 w:     the output result, which is an approximation of exp(t A) v
 error: the error between w and actual exp(t A) v, ||w - exp(t A) v||
 hump:  hump = max(exp(s A)), s in [0, t]. It is used as a measure of conditioning of the matrix exponetial. (Please refer the paper for more detailed explaination)
 
 @return:
 whether the process succeed. True for yes, false for error.
*/
{
    const double PI = 3.1415926;
    assert(A.rows() == A.cols());
    int n = A.rows();
    
    double anorm = A.norm();
    double mxrej = 10; double btol = 1e-7;
    double gamma = 0.9; double delta = 1.2;
    int mb = m; double t_out = std::abs(t);
    int nstep = 0; double t_new = 0;
    double t_now = 0; double s_error = 0;
    double rndoff = anorm * 1e-16;
    
    int k1 = 2; double xm = 1.0 / m; double normv = v.norm(); double beta = normv;
    double fact = std::pow((m+1) / std::exp(1), m + 1) * std::sqrt(2 * PI * (m + 1));
    t_new = (1 / anorm) * std::pow((fact * tol) / (4 * beta * anorm), xm);
    double s = std::pow(10.0, std::floor(std::log10(t_new)) - 1); t_new = std::ceil(t_new/s) * s;
    int sgn = t >= 0? 1 : -1; nstep = 0;
    
    w = v;
    double mhump = normv;
    int mx = 0;
    double err_loc = 0;
    
    Eigen::MatrixXd F;
    
    while (t_now < t_out)
    {
        nstep = nstep + 1;
        double t_step = std::min(t_out - t_now, t_new);
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n,m + 1);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m + 2, m + 2);
        
        V.col(0) = (1.0 / beta) * w;
        
        for(int j = 0 ; j < m ;j++)
        {
            Eigen::VectorXd p = A*V.col(j);
            for (int i=0; i < j;i++)
            {
                H(i,j) = V.col(i).dot(p);
                p = p - H(i,j) * V.col(i);
            }
            s = p.norm();
            if(s < btol)
            {
                k1 = 0;
                mb = j;
                t_step = t_out - t_now;
                break;
            }
            H(j + 1,j) = s;
            V.col(j+1) = (1.0 / s) * p;
        }
        double avnorm = 0;
        if (k1 != 0)
        {
            H(m + 1,m) = 1;
            avnorm = (A*V.col(m)).norm();
        }
        double ireject = 0;
        while (ireject <= mxrej)
        {
            mx = mb + k1;
            Eigen::MatrixXd tmpH =sgn*t_step*H.block(0, 0, mx, mx);
            F = tmpH.exp(); // according to Eigen's documentation "https://eigen.tuxfamily.org/dox/unsupported/group__MatrixFunctions__Module.html", the computation is approximately 20 * mx^3
            if(k1 == 0)
            {
                err_loc = btol;
                break;
            }
            else
            {
                double phi1 = abs(beta * F(m, 1));
                double phi2 = abs(beta * F(m+1, 1) * avnorm);
                if(phi1 > 10*phi2)
                {
                    err_loc = phi2;
                    xm = 1/m;
                }
                else if(phi1 > phi2)
                {
                    err_loc = (phi1*phi2)/(phi1-phi2);
                    xm = 1/m;
                }
                else
                {
                    err_loc = phi1;
                    xm = 1/(m-1);
                }
            }
               
            if(err_loc <= delta * t_step*tol)
               break;
            else
            {
                t_step = gamma * t_step * std::pow(t_step*tol/err_loc, xm);
                s = std::pow(10.0, floor(log10(t_step))-1);
                t_step = std::ceil(t_step/s) * s;
                if(ireject == mxrej)
                {
                    std::cerr << "The requested tolerance is too high." << std::endl;
                    return false;
                }
                ireject = ireject + 1;
            }
        }
        mx = mb + std::max(0, k1 - 1);
        w = V.block(0, 0, V.rows(), mx) * (beta * F.block(0, 0, mx, 1));
        beta = w.norm();
        mhump = std::max(mhump,beta);

        t_now = t_now + t_step;
        t_new = gamma * t_step * std::pow(t_step*tol/err_loc, xm);
        s = std::pow(10.0, std::floor(log10(t_new))-1);
        t_new = std::ceil(t_new/s) * s;

        err_loc = std::max(err_loc,rndoff);
        s_error = s_error + err_loc;
    }
    
    if(err)
        (*err) = s_error;
    if(hump)
        (*hump) = mhump / normv;
    return true;
} 
