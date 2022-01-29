#include "CtPDController.h"
#include "sim/SimCharacter.h"

#include <unistd.h>
#include <iostream>

const std::string gPDControllersKey = "PDControllers";

cCtPDController::cCtPDController() : cCtController()
{
	mGravity = gGravity;
}

cCtPDController::~cCtPDController()
{
}

void cCtPDController::Reset()
{
	cCtController::Reset();
	mPDCtrl.Reset();
}

void cCtPDController::Clear()
{
	cCtController::Clear();
	mPDCtrl.Clear();
}

void cCtPDController::SetGravity(const tVector& gravity)
{
	mGravity = gravity;
}

std::string cCtPDController::GetName() const
{
	return "ct_pd";
}

void cCtPDController::SetupPDControllers(const Json::Value& json, const tVector& gravity)
{
	Eigen::MatrixXd pd_params;
	bool succ = false;

	if (!json[gPDControllersKey].isNull())
	{
		succ = cPDController::LoadParams(json[gPDControllersKey], pd_params);
	}

	if (succ)
	{
		mPDCtrl.Init(mChar, pd_params, gravity);
	}

	mValid = succ;
	if (!mValid)
	{
		printf("Failed to initialize Ct-PD controller\n");
		mValid = false;
	}

}

bool cCtPDController::ParseParams(const Json::Value& json)
{
	bool succ = cCtController::ParseParams(json);
	SetupPDControllers(json, mGravity);
	return succ;
}

void cCtPDController::UpdateBuildTau(double time_step, Eigen::VectorXd& out_tau)
{
	UpdatePDCtrls(time_step, out_tau);
}

void cCtPDController::UpdatePDCtrls(double time_step, Eigen::VectorXd& out_tau)
{
	int num_dof = mChar->GetNumDof();
	out_tau = Eigen::VectorXd::Zero(num_dof);
	mPDCtrl.UpdateControlForce(time_step, out_tau);
}

void cCtPDController::ApplyAction(const Eigen::VectorXd& action)
{
	cCtController::ApplyAction(action);
	SetPDTargets(action);
}

void cCtPDController::BuildJointActionBounds(int joint_id, Eigen::VectorXd& out_min, Eigen::VectorXd& out_max) const
{
	const Eigen::MatrixXd& joint_mat = mChar->GetJointMat();
	cCtCtrlUtil::BuildBoundsPD(joint_mat, joint_id, out_min, out_max);
}

void cCtPDController::BuildJointActionOffsetScale(int joint_id, Eigen::VectorXd& out_offset, Eigen::VectorXd& out_scale) const
{
	const Eigen::MatrixXd& joint_mat = mChar->GetJointMat();
	cCtCtrlUtil::BuildOffsetScalePD(joint_mat, joint_id, out_offset, out_scale);
}

void cCtPDController::ConvertActionToTargetPose(int joint_id, Eigen::VectorXd& out_theta) const
{
#if defined(ENABLE_PD_SPHERE_AXIS)
	cKinTree::eJointType joint_type = GetJointType(joint_id);
	if (joint_type == cKinTree::eJointTypeSpherical)
	{
		double rot_theta = out_theta[0];
		tVector axis = tVector(out_theta[1], out_theta[2], out_theta[3], 0);
		if (axis.squaredNorm() == 0)
		{
			axis[2] = 1;
		}

		axis.normalize();
		tQuaternion quat = cMathUtil::AxisAngleToQuaternion(axis, rot_theta);

		if (FlipStance())
		{
			cKinTree::eJointType joint_type = GetJointType(joint_id);
			if (joint_type == cKinTree::eJointTypeSpherical)
			{
				quat = cMathUtil::MirrorQuaternion(quat, cMathUtil::eAxisZ);
			}
		}
		out_theta = cMathUtil::QuatToVec(quat);
	}
#endif
}

cKinTree::eJointType cCtPDController::GetJointType(int joint_id) const
{
	const cPDController& ctrl = mPDCtrl.GetPDCtrl(joint_id);
	const cSimBodyJoint& joint = ctrl.GetJoint();
	cKinTree::eJointType joint_type = joint.GetType();
	return joint_type;
}

void cCtPDController::SetPDTargets(const Eigen::VectorXd& targets)
{
	int root_id = mChar->GetRootID();
	int root_size = mChar->GetParamSize(root_id);
	int num_joints = mChar->GetNumJoints();
	int ctrl_offset = GetActionCtrlOffset();

	const double time = mKinChar->GetTime();
	Eigen::VectorXd kin_pose, kin_vel;
	mKinChar->CalcPose(time, kin_pose); //Angular position in quaternion
	//mKinChar->CalcVel(time, kin_vel); //Angular velocity in quaternion

	for (int j = root_id + 1; j < num_joints; ++j)
	{
		if (mPDCtrl.IsValidPDCtrl(j))
		{
			int retarget_joint = RetargetJointID(j);
			int param_offset = mChar->GetParamOffset(retarget_joint);
			int param_size = mChar->GetParamSize(retarget_joint);
			param_offset -= root_size;
			param_offset += ctrl_offset;
			Eigen::VectorXd theta = targets.segment(param_offset, param_size);
			
			ConvertActionToTargetPose(j, theta);
			
			if (param_size > 0){
				AddPoseOffset(j, theta, kin_pose);
			}
			
			mPDCtrl.SetTargetTheta(j, theta);

			//Eigen::VectorXd vel = kin_vel.segment(param_offset, param_size);
			//ConvertActionToTargetPose(j, vel);
			//mPDCtrl.SetTargetVel(j, vel);
		}
	}
}

void cCtPDController::AddPoseOffset(int j, Eigen::VectorXd& theta, Eigen::VectorXd& pose)
{
	tQuaternion tar_q = cMathUtil::VecToQuat(theta);
	const auto& joint_mat = mKinChar->GetJointMat();
	int param_offset = mKinChar->GetParamOffset(j);
	int param_size = mKinChar->GetParamSize(j);
	tQuaternion kin_q = cMathUtil::VecToQuat(pose.segment(param_offset, param_size));
	//tar_q = kin_q;
	tar_q = kin_q * tar_q;
	theta = cMathUtil::QuatToVec(tar_q.normalized());
}

void cCtPDController::SetKinChar(std::shared_ptr<cKinCharacter> kin_char)
{
	mPDCtrl.SetKinChar(kin_char);
	mKinChar = kin_char;
}

void cCtPDController::RecordPose(Eigen::VectorXd& out_pose) const
{	
	const Eigen::VectorXd& pose0 = mChar->GetPose();	
	const Eigen::VectorXd& pose1 = mKinChar->GetPose();

	int pose0_size = int(pose0.size());
	int pose1_size = int(pose1.size());
	// fill with nans to make sure we don't forget to set anything
	out_pose = std::numeric_limits<double>::quiet_NaN() * Eigen::VectorXd::Ones(pose0_size + pose1_size);
	
	out_pose.segment(0, pose0_size) = pose0;
	out_pose.segment(pose0_size, pose1_size) = pose1;
}

void cCtPDController::BuildStatePose(Eigen::VectorXd& out_pose) const
{
	cCtController::BuildStatePose(out_pose);
	//std::cerr << mKinChar->GetPose()[0] << std::endl;

	int offset = cCtController::GetStatePoseSize();
	auto pose = mKinChar->GetPose();
	//std::cerr << offset << " " << pose << std::endl; //changing depend on time

	tMatrix origin_trans = mKinChar->BuildOriginTrans();
	tQuaternion origin_quat = cMathUtil::RotMatToQuaternion(origin_trans);

	bool flip_stance = FlipStance();
	if (flip_stance)
	{
		origin_trans.row(2) *= -1; // reflect z
	}

	tVector root_pos = mKinChar->GetRootPos();
	tVector root_pos_rel = root_pos;

	root_pos_rel[3] = 1;
	root_pos_rel = origin_trans * root_pos_rel;
	root_pos_rel[3] = 0;

	out_pose[offset] = root_pos_rel[1];

	const auto& joint_mat = mChar->GetJointMat();
	const auto& body_defs = mChar->GetBodyDefs();

	int num_parts = cKinTree::GetNumJoints(joint_mat);
	int root_id = cKinTree::GetRoot(joint_mat);

	int pos_dim = GetPosFeatureDim();
	int rot_dim = GetRotFeatureDim();

	tQuaternion mirror_inv_origin_quat = origin_quat.conjugate();
	mirror_inv_origin_quat = cMathUtil::MirrorQuaternion(mirror_inv_origin_quat, cMathUtil::eAxisZ);

	int idx = 1+offset;
	for (int i = 0; i < num_parts; ++i)
	{
		int part_id = RetargetJointID(i);
		if (mChar->IsValidBodyPart(part_id))
		{
			tVector curr_pos = cKinTree::CalcBodyPartPos(joint_mat, body_defs, pose, i);

			if (mRecordWorldRootPos && i == root_id)
			{
				if (flip_stance)
				{
					curr_pos = cMathUtil::QuatRotVec(origin_quat, curr_pos);
					curr_pos[2] = -curr_pos[2];
					curr_pos = cMathUtil::QuatRotVec(mirror_inv_origin_quat, curr_pos);
				}
			}
			else
			{
				curr_pos[3] = 1;
				curr_pos = origin_trans * curr_pos;
				curr_pos -= root_pos_rel;
				curr_pos[3] = 0;
			}

			out_pose.segment(idx, pos_dim) = curr_pos.segment(0, pos_dim);
			idx += pos_dim;

			tVector axis;
			double angle;
			cKinTree::CalcBodyPartRotation(joint_mat, body_defs, pose, i, axis, angle);
			tQuaternion curr_quat = cMathUtil::AxisAngleToQuaternion(axis, angle);
			if (mRecordWorldRootRot && i == root_id)
			{
				if (flip_stance)
				{
					curr_quat = origin_quat * curr_quat;
					curr_quat = cMathUtil::MirrorQuaternion(curr_quat, cMathUtil::eAxisZ);
					curr_quat = mirror_inv_origin_quat * curr_quat;
				}
			}
			else
			{
				curr_quat = origin_quat * curr_quat;
				if (flip_stance)
				{
					curr_quat = cMathUtil::MirrorQuaternion(curr_quat, cMathUtil::eAxisZ);
				}
			}

			if (curr_quat.w() < 0)
			{
				curr_quat.w() *= -1;
				curr_quat.x() *= -1;
				curr_quat.y() *= -1;
				curr_quat.z() *= -1;
			}
			out_pose.segment(idx, rot_dim) = cMathUtil::QuatToVec(curr_quat).segment(0, rot_dim);
			idx += rot_dim;
		}
	}
}

int cCtPDController::GetStatePoseSize() const
{
	return cCtController::GetStatePoseSize() * 2; 
}

void cCtPDController::BuildStateVel(Eigen::VectorXd& out_vel) const
{
	cCtController::BuildStateVel(out_vel);
	int offset = cCtController::GetStateVelSize();
	auto pose = mKinChar->GetPose();
	auto vel = mKinChar->GetVel();

	const auto& joint_mat = mChar->GetJointMat();
	const auto& body_defs = mChar->GetBodyDefs();

	int num_parts = cKinTree::GetNumJoints(joint_mat);
	tMatrix origin_trans = mKinChar->BuildOriginTrans();
	tQuaternion origin_quat = cMathUtil::RotMatToQuaternion(origin_trans);

	bool flip_stance = FlipStance();
	if (flip_stance)
	{
		origin_trans.row(2) *= -1; // reflect z
	}

	int pos_dim = GetPosFeatureDim();
	int rot_dim = GetRotFeatureDim();

	tQuaternion mirror_inv_origin_quat = origin_quat.conjugate();
	mirror_inv_origin_quat = cMathUtil::MirrorQuaternion(mirror_inv_origin_quat, cMathUtil::eAxisZ);
	
	int idx = offset;
	for (int i = 0; i < num_parts; ++i)
	{
		int part_id = RetargetJointID(i);
		int root_id = cKinTree::GetRoot(joint_mat);

		tVector curr_vel = cKinTree::CalcBodyPartVel(joint_mat, body_defs, pose, vel, i);
		if (mRecordWorldRootRot && i == root_id)
		{
			if (flip_stance)
			{
				curr_vel = cMathUtil::QuatRotVec(origin_quat, curr_vel);
				curr_vel[2] = -curr_vel[2];
				curr_vel = cMathUtil::QuatRotVec(mirror_inv_origin_quat, curr_vel);
			}
		}
		else
		{
			curr_vel = origin_trans * curr_vel;
		}

		out_vel.segment(idx, pos_dim) = curr_vel.segment(0, pos_dim);
		idx += pos_dim;

		tVector curr_ang_vel = cKinTree::CalcJointWorldAngularVel(joint_mat, pose, vel, i);
		if (mRecordWorldRootRot && i == root_id)
		{
			if (flip_stance)
			{
				curr_ang_vel = cMathUtil::QuatRotVec(origin_quat, curr_ang_vel);
				curr_ang_vel[2] = -curr_ang_vel[2];
				curr_ang_vel = -curr_ang_vel;
				curr_ang_vel = cMathUtil::QuatRotVec(mirror_inv_origin_quat, curr_ang_vel);
			}
		}
		else
		{
			curr_ang_vel = origin_trans * curr_ang_vel;
			if (flip_stance)
			{
				curr_ang_vel = -curr_ang_vel;
			}
		}

		out_vel.segment(idx, rot_dim - 1) = curr_ang_vel.segment(0, rot_dim - 1);
		idx += rot_dim - 1;
	}
}

int cCtPDController::GetStateVelSize() const
{
	return cCtController::GetStateVelSize() * 2;
}