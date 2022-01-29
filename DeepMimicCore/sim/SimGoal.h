#include <memory>
#include "sim/SimRigidBody.h"

class cSimGoal
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	cSimGoal();
	virtual ~cSimGoal();

	virtual void Init();
	//virtual void Reset(const tVector& pos, const double radius, const int flag);
	virtual int GetSize() const;
	virtual void SetPos(const Eigen::VectorXd& pos);
	virtual void SetRadius(const double& radius);
	virtual void SetFlagClear(const int& Flag);
	virtual void SetFlagTime(const int& Flag);
	virtual Eigen::VectorXd GetPos() const;
	virtual int GetFlagClear() const;
	virtual int GetFlagTime() const;
	virtual double GetRadius() const;
	///virtual tVector GetSize() const;

	virtual cShape::eShape GetShape() const;

	
protected:
	tVector mPos;
    double mRadius;
    int mFlagClear;
	int mFlagTime;

};