#ifndef SCENEOBJECTS_H
#define SCENEOBJECTS_H

#include <Eigen/Core>
#include "SimParameters.h"
#include <set>

struct Particle
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Particle(Eigen::Vector2d pos, double mass, bool isFixed, bool isInert) : pos(pos), mass(mass), fixed(isFixed), inert(isInert)
    {
        vel.setZero();
        prevpos = pos;
    }

    Eigen::Vector2d pos;
    Eigen::Vector2d prevpos;
    Eigen::Vector2d vel;
    Eigen::Vector2d preVel;
    double mass;
    bool fixed;
    bool inert;
};


struct Connector1d // connect particle to the ceil
{
public:
    Connector1d(int p, double mass) : p(p), mass(mass) {}
    virtual ~Connector1d() {}

    int p;
    double mass;

    std::set<int> associatedBendingStencils;
};

struct Spring1d : public Connector1d
{
public:
    Spring1d(int p, double mass, double stiffness, double restDis, bool canSnap) : Connector1d(p, mass), stiffness(stiffness), restDis(restDis), canSnap(canSnap) {}

    double stiffness;
    double restDis;
    bool canSnap;
};


struct Connector
{
public:
    Connector(int p1,int p2, double mass) : p1(p1), p2(p2), mass(mass) {}
    virtual ~Connector() {}

    int p1;
    int p2;
    double mass;

    std::set<int> associatedBendingStencils;
};

struct Spring : public Connector
{
public:
    Spring(int p1, int p2, double mass, double stiffness, double restlen, bool canSnap) : Connector(p1, p2, mass), stiffness(stiffness), restlen(restlen), canSnap(canSnap) {}

    double stiffness;
    double restlen;
    bool canSnap;
};

struct Saw
{
public:
    Saw(Eigen::Vector2d pos, double radius) : pos(pos), radius(radius) {}

    Eigen::Vector2d pos;
    double radius;
};

#endif