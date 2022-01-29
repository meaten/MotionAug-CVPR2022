#pragma once

#include "sim/CtPDController.h"

class cCtPDResController : public virtual cCtPDController
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	enum eResForceType
	{
		eResForceZero,
		eResForceScratch,
		eResForcePD,
		eResForcePD2,
		eResForceRefine
	};
	
	cCtPDResController();
	virtual ~cCtPDResController();

	virtual std::string GetName() const;
	virtual Eigen::VectorXd GetResForce();
protected:
	Eigen::VectorXd mResForce;
	eResForceType mResForceType;

	bool ParseParams(const Json::Value& json);

	virtual void ApplyAction(const Eigen::VectorXd& action);
	virtual int GetActionSize() const;
	virtual int GetActionResForceSize() const;
	virtual void UpdateApplyTau(const Eigen::VectorXd& tau);
	virtual void RootPD(Eigen::VectorXd& out_resforce);
	void CalcResForce(tVector& force) const;
	void CalcResTorque(tVector& torque) const;

	double GetPhase() const;

	void BuildActionBounds(Eigen::VectorXd& out_min, Eigen::VectorXd& out_max) const;
};