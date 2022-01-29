#include "SimGoal.h"

cSimGoal::cSimGoal()
{
}

cSimGoal::~cSimGoal()
{
}

void cSimGoal::Init()
{
	mPos = tVector(0,0,0,0);
	mRadius = 0.2;
	mFlagClear = 0;
	mFlagTime = 0;
}
/*
void cSimGoal::Reset(const tVector& pos, const double radius, const int flag) 
{
	SetPos(pos);
	SetRadius(radius);
	SetFlag(flag);
}*/

int cSimGoal::GetSize() const
{
	return 5;
}

void cSimGoal::SetPos(const Eigen::VectorXd& pos) 
{
	mPos = tVector(pos);
}

void cSimGoal::SetRadius(const double& radius) 
{
	mRadius = radius;
}

void cSimGoal::SetFlagClear(const int& Flag) 
{
	mFlagClear = Flag;
}
void cSimGoal::SetFlagTime(const int& Flag) 
{
	mFlagTime = Flag;
}

Eigen::VectorXd cSimGoal::GetPos() const
{
	return mPos;
}

int cSimGoal::GetFlagClear() const
{
	return mFlagClear;
}


int cSimGoal::GetFlagTime() const
{
	return mFlagTime;
}

double cSimGoal::GetRadius() const
{
	return mRadius;
}
/*
tVector cSimGoal::GetSize() const
{
	double r = mRadius;
	double scale = mWorld->GetScale();
	r /= scale;
	double d = 2 * r;
	return tVector(d, d, d, 0);
}
*/
cShape::eShape cSimGoal::GetShape() const
{
	return cShape::eShapeSphere;
}
