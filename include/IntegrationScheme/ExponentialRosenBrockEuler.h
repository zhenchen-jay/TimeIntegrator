#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "../Utils/MatrixVecExp.h"

namespace TimeIntegrator
{
    /*
    * For the system we aimed to solve:
    * dx / dt = v
    * M dv / dt = F(x)
    * where F(x) = -\nabla E(x),
    * E(x) = 1/2 x^T M x + potential_energy(x).
    *
    *
    Exponetial RosenBrock Euler:
    d[x, v]^T / dt = [v, M^{-1} F]^T
    Let u = [x, v]^T, and b(u) = [v, M{-1} F]^T, then
    du / dt = b(u(t)).

    Followed by the formula given in "https://www.cs.ubc.ca/~ascher/papers/cap.pdf", we have
    u_{n+1} = [I_N, 0_{N x 1}] exp(h A) ubar_n, where
    ubar_n = [u_n, 1]^T
    A <<
    J_n = db / du(u_n), c(u_n) = b(u_n) - J_n u_n
    o_{1 x N}, 0
    */

    template <typename Problem>
    void exponentialRosenBrockEuler(const Eigen::VectorXd& xcur, const Eigen::VectorXd& vcur, const double h, const Eigen::VectorXd& M, Problem energyModel, Eigen::VectorXd& xnext, Eigen::VectorXd& vnext)
    {
        if(h == 0)
        {
            xnext = xcur;
            vnext = vcur;
        }
        Eigen::VectorXd un(xcur.size() + vcur.size()), unbar(xcur.size() + vcur.size() + 1);
        un.segment(0, xcur.size()) = xcur;
        un.segment(xcur.size(), vcur.size()) = vcur;

        unbar.segment(0, un.size()) = un;
        unbar(un.size()) = 1.0;

        // gradient and hessian
        Eigen::VectorXd grad;
        energyModel.computeGradient(xcur, grad);
        Eigen::SparseMatrix<double> H;
        energyModel.computeHessian(xcur, H);

        // mass inverse matrix
        std::vector<Eigen::Triplet<double>> massTrip;
        Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

        for (int i = 0; i < M.size(); i++)
            massTrip.push_back(Eigen::Triplet<double>(i, i, M(i)));
        massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());

        Eigen::SparseMatrix<double> MInvH = massMatInv * H, Jn(un.size(), un.size()), A(un.size() + 1, un.size() + 1);

        // Jn, A, and [I, 0]
        std::vector<Eigen::Triplet<double>> Jlist, Alist, IzeroList;

        for (int i = 0; i < vcur.size(); i++)
        {
            Jlist.push_back(Eigen::Triplet<double>(i, xcur.size() + i, 1.0));
        }

        for (int k = 0; k < MInvH.outerSize(); k++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(MInvH, k); it; ++it) {
                Jlist.push_back(Eigen::Triplet<double>(vcur.rows() + it.row(), it.col(), -it.value()));
            }
        }

        Jn.setFromTriplets(Jlist.begin(), Jlist.end());
        Eigen::VectorXd bn(un.size()), cn(un.size());

        bn.segment(0, vcur.size()) = vcur;
        bn.segment(vcur.size(), xcur.size()) = -grad;

        cn = bn - Jn * un;

        for (int k = 0; k < Jn.outerSize(); k++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Jn, k); it; ++it) {
                Alist.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
            }
        }

        for (int i = 0; i < cn.size(); i++)
            Alist.push_back(Eigen::Triplet<double>(un.size(), i, cn(i)));
        Alist.push_back(Eigen::Triplet<double>(un.size(), un.size(), 1.0));

        A.setFromTriplets(Alist.begin(), Alist.end());

        Eigen::SparseMatrix<double> IzeroMat(un.size(), unbar.size());
        for (int i = 0; i < un.size(); i++)
            IzeroList.push_back(Eigen::Triplet<double>(i, i, 1));
        IzeroMat.setFromTriplets(IzeroList.begin(), IzeroList.end());



        Eigen::VectorXd unplus1;
        matrixVecExp<Eigen::SparseMatrix<double>>(A, unbar, h, 30, 1e-9, unplus1);

        unplus1 = IzeroMat * unplus1;
        xnext = unplus1.segment(0, xcur.size());
        vnext = unplus1.segment(xcur.size(), vcur.size());

    }
}

