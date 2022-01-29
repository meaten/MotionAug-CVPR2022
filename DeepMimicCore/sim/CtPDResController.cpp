#include "CtPDResController.h"
#include "sim/SimCharacter.h"
#include "util/MathUtil.h"

#include <unistd.h>
#include <iostream>

const std::string gPDControllersKey = "PDControllers";
const std::string gResForceTypeKey = "ResForceType";

cCtPDResController::cCtPDResController() : cCtPDController()
{
	Eigen::VectorXd mResForce(6);
	mResForce << 0,0,0,0,0,0;

	mResForceType = eResForceType::eResForceScratch;
}

cCtPDResController::~cCtPDResController()
{
}

bool cCtPDResController::ParseParams(const Json::Value& json)
{
	bool succ = cCtPDController::ParseParams(json);
	std::string type = "rootPD_weight_1";
	type = json.get(gResForceTypeKey, type).asString();
	if (type == "zero"){
		mResForceType = eResForceZero;
	} else if(type == "scratch"){
		mResForceType = eResForceType::eResForceScratch;
	} else if (type == "rootPD"){
		mResForceType = eResForceType::eResForcePD;
	} else if (type == "rootPD_weight_1"){
		mResForceType = eResForceType::eResForcePD2;
	} else if (type == "rootPD_refine"){
		mResForceType = eResForceType::eResForceRefine;
	} else {
		throw std::invalid_argument("bad resforce type");
	}

	std::cerr << type << std::endl;

	return succ;
}

std::string cCtPDResController::GetName() const
{
	return "ct_pd_res";
}

double cCtPDResController::GetPhase() const
{
	double phase = mTime / mKinChar->GetMotionDuration();
	phase += mPhaseOffset;
	phase = std::fmod(phase, 1.0);
	phase = (phase < 0) ? (1 + phase) : phase;
	return phase;
}

void cCtPDResController::ApplyAction(const Eigen::VectorXd& action)
{
	assert(action.size() == GetActionSize());

	cCtController::ApplyAction(action);
	Eigen::VectorXd action_theta;
	mResForce = Eigen::VectorXd::Zero(6);
	
	if (mResForceType == eResForceScratch){
		mResForce += action.tail(GetActionResForceSize()) * 100;	
	} else if(mResForceType == eResForcePD){
		RootPD(mResForce);
		double coef_action = action.tail(GetActionResForceSize())[0];
		mResForce *= coef_action;
	} else if(mResForceType == eResForcePD2){
		RootPD(mResForce);
	} else if(mResForceType == eResForceRefine){
		RootPD(mResForce);
		mResForce += action.tail(GetActionResForceSize()) * 100;
	}

	action_theta = action.head(GetActionSize() - GetActionResForceSize());
	SetPDTargets(action_theta);
}

void cCtPDResController::RootPD(Eigen::VectorXd& out_resforce)
{
	tVector force, torque;
	CalcResForce(force);
	CalcResTorque(torque);
	out_resforce << force[0], force[1], force[2], torque[0], torque[1], torque[2];
}

void cCtPDResController::CalcResForce(tVector& force) const
{
	tVector pos_err = mKinChar->CalcJointPos(0) - mChar->CalcJointPos(0);
	tVector vel_err = mKinChar->CalcJointVel(0) - mChar->CalcJointVel(0);
	force = pos_err * 200 + vel_err * 100;
	//force *= 0;
}

void cCtPDResController::CalcResTorque(tVector& torque) const
{
	tVector out_axis;
	double angle;

	tQuaternion quat_ang_err = cMathUtil::QuatDiff(mChar->CalcJointWorldRotation(0), mKinChar->CalcJointWorldRotation(0));
	cMathUtil::QuaternionToAxisAngle(quat_ang_err, out_axis, angle);
	tVector ang_err = out_axis * angle;
	
	tVector kin_vel = mKinChar->GetRootAngVel(); //axis_angle
	tVector char_vel = mChar->GetRootAngVel();
	tVector vel_err = kin_vel - char_vel;

	torque = ang_err * 100 + vel_err * 10;

	//cMathUtil::QuaternionToAxisAngle(mChar->CalcJointWorldRotation(0), out_axis, angle);
	//std::cerr << mKinChar->GetRootAngVel() << std::endl << std::endl << mChar->GetRootAngVel() << std::endl << std::endl << out_axis * angle << std::endl << std::endl;
}

void cCtPDResController::UpdateApplyTau(const Eigen::VectorXd& tau)
{
	mTau = tau;
	mChar->ApplyControlForces(tau, mResForce);
}

int cCtPDResController::GetActionSize() const
{
	return cCtPDController::GetActionSize() + GetActionResForceSize();
}

int cCtPDResController::GetActionResForceSize() const
{	
	if (mResForceType == eResForceZero){
		return 0;
	} else if (mResForceType == eResForceScratch){
		return 6;
	} else if(mResForceType == eResForcePD){
		return 1;
	} else if(mResForceType == eResForcePD2){
		return 0;
	} else if(mResForceType == eResForceRefine){
		return 6;
	}
}

Eigen::VectorXd cCtPDResController::GetResForce()
{
	return mResForce;
}

void cCtPDResController::BuildActionBounds(Eigen::VectorXd& out_min, Eigen::VectorXd& out_max) const
{	
	cCtPDController::BuildActionBounds(out_min, out_max);

	int param_offset = cCtPDController::GetActionSize();
	int param_size = GetActionResForceSize();
	if (param_size == 0){
		return;
	}
	if (mResForceType == eResForceScratch){
		Eigen::VectorXd lim_min = Eigen::VectorXd::Ones(param_size);
		Eigen::VectorXd lim_max = Eigen::VectorXd::Ones(param_size);
		lim_min *= -10;
		lim_max *= 10;
		out_min.segment(param_offset, param_size) = lim_min;
		out_max.segment(param_offset, param_size) = lim_max;
	} else if(mResForceType == eResForcePD){
		out_min[param_offset] = 0;
		out_max[param_offset] = 1;
	} else if(mResForceType == eResForcePD2){
		//we cannot reach this
	}
}
