#pragma once

#include "sim/CtController.h"
#include "sim/ImpPDController.h"

class cCtPDController : public virtual cCtController
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	cCtPDController();
	virtual ~cCtPDController();

	virtual void Reset();
	virtual void Clear();

	virtual void SetGravity(const tVector& gravity);

	virtual void RecordPose(Eigen::VectorXd& out_pose) const;

	virtual std::string GetName() const;

protected:
	cImpPDController mPDCtrl;
	std::shared_ptr<cKinCharacter> mKinChar;

	tVector mGravity;

	virtual bool ParseParams(const Json::Value& json);

	virtual void UpdateBuildTau(double time_step, Eigen::VectorXd& out_tau);
	virtual void SetupPDControllers(const Json::Value& json, const tVector& gravity);
	virtual void UpdatePDCtrls(double time_step, Eigen::VectorXd& out_tau);
	virtual void ApplyAction(const Eigen::VectorXd& action);
	virtual void BuildJointActionBounds(int joint_id, Eigen::VectorXd& out_min, Eigen::VectorXd& out_max) const;
	virtual void BuildJointActionOffsetScale(int joint_id, Eigen::VectorXd& out_offset, Eigen::VectorXd& out_scale) const;
	virtual void ConvertActionToTargetPose(int joint_id, Eigen::VectorXd& out_theta) const;
	virtual cKinTree::eJointType GetJointType(int joint_id) const;

	virtual void SetPDTargets(const Eigen::VectorXd& targets);

	virtual void SetKinChar(std::shared_ptr<cKinCharacter> kin_char);
	virtual void BuildStatePose(Eigen::VectorXd& out_pose) const;
	virtual void BuildStateVel(Eigen::VectorXd& out_vel) const;
	virtual int GetStatePoseSize() const;
	virtual int GetStateVelSize() const;

	virtual void AddPoseOffset(int j, Eigen::VectorXd& theta, Eigen::VectorXd& pose);
};