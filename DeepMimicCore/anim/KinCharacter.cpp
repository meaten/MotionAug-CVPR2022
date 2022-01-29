#include "KinCharacter.h"
#include <assert.h>
#include <functional>
#include "util/FileUtil.h"

#include <unistd.h>
#include <iostream>

#include "util/FindPeaks.h"

const double gDiffTimeStep = 1 / 600.0;

cKinCharacter::tParams::tParams()
{
	mID = gInvalidIdx;
	mCharFile = "";
	mMotionFile = "";
	mMotionString = "";
	mOrigin.setZero();
	mLoadDrawShapes = true;
	mCenter = {};
	mGoalPos = {};
	mIKJoint = {};
	mPertitionTime = {};
}

cKinCharacter::cKinCharacter()
{
	mOrigin.setZero();
	mOriginRot.setIdentity();
	mCycleRootDelta.setZero();
}

cKinCharacter::~cKinCharacter()
{
}

bool cKinCharacter::Init(const tParams& params)
{
	mID = params.mID;
	bool succ = cCharacter::Init(params.mCharFile, params.mLoadDrawShapes);
	if (succ)
	{
		if (params.mMotionString != ""){
			LoadMotion(params.mMotionString);
		} else if (params.mMotionFile != "")
		{
			LoadMotion(params.mMotionFile, params.mGoalPos, params.mCenter, params.mIKJoint, params.mPertitionTime, params.mWindowTime);
		}

		if (params.mStateFile != "")
		{
			bool succ_state = ReadState(params.mStateFile);

			if (!succ_state)
			{
				printf("Failed to load character state from %s\n", params.mStateFile.c_str());
			}
			else
			{
				mPose0 = mPose;
				mVel0 = mVel;
				SetPose(mPose);
				SetVel(mVel);
			}
		}

		SetOriginPos(params.mOrigin);
	}
	else
	{
		printf("Failed to build character from char_file: %s\n", params.mCharFile.c_str());
	}
	return succ;
}

void cKinCharacter::Clear()
{
	cCharacter::Clear();
	mMotion.Clear();
}

void cKinCharacter::Update(double time_step)
{
	cCharacter::Update(time_step);
	mTime += time_step;
	Pose(mTime);
}

void cKinCharacter::Reset()
{
	cCharacter::Reset();
}

bool cKinCharacter::LoadMotion(const std::string& motion_file, 
							   const std::vector<tVector> GoalPos, 
							   const std::vector<tVector> Center, 
							   const std::vector<int> IKJoint,
							   const std::vector<double> PertitionTime,
							   const double WindowTime)
{
	cMotion::tParams motion_params;
	motion_params.mMotionFile = motion_file;
	motion_params.mBlendFunc = std::bind(&cKinCharacter::BlendFrames, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
											std::placeholders::_4);
	motion_params.mMirrorFunc = std::bind(&cKinCharacter::MirrorFrame, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	motion_params.mVelFunc = std::bind(&cKinCharacter::CalcFrameVel, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
											std::placeholders::_4);
	motion_params.mPostProcessFunc = std::bind(&cKinCharacter::PostProcessFrame, this, std::placeholders::_1);
	motion_params.mGoalPos = GoalPos;
	motion_params.mCenter = Center;
	motion_params.mIKJoint = IKJoint;
	motion_params.mPertitionTime = PertitionTime;
	motion_params.mWindowTime = WindowTime;

	bool succ = mMotion.Load(motion_params);

	if (succ)
	{
		int char_dof = GetNumDof();
		int motion_dof = mMotion.GetNumDof();

		if (char_dof != motion_dof)
		{
			printf("DOF mismatch, char dof: %i, motion dof: %i\n", char_dof, motion_dof);
			mMotion.Clear();
			succ = false;
		}
	}

	if (succ)
	{
		mCycleRootDelta = CalcCycleRootDelta();
		Pose(mTime);
		mPose0 = GetPose();
		mVel0 = GetVel();
	}

	if (!succ)
	{
		printf("Failed to load motion from: %s\n", motion_file.c_str());
	}
	return succ;
}

bool cKinCharacter::LoadMotion(const std::string& motion_string)
{
	cMotion::tParams motion_params;
	motion_params.mMotionString = motion_string;
	motion_params.mBlendFunc = std::bind(&cKinCharacter::BlendFrames, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
											std::placeholders::_4);
	motion_params.mMirrorFunc = std::bind(&cKinCharacter::MirrorFrame, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	motion_params.mVelFunc = std::bind(&cKinCharacter::CalcFrameVel, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
											std::placeholders::_4);
	motion_params.mPostProcessFunc = std::bind(&cKinCharacter::PostProcessFrame, this, std::placeholders::_1);
	bool succ = mMotion.Load(motion_params);

	if (succ)
	{
		int char_dof = GetNumDof();
		int motion_dof = mMotion.GetNumDof();

		if (char_dof != motion_dof)
		{
			printf("DOF mismatch, char dof: %i, motion dof: %i\n", char_dof, motion_dof);
			mMotion.Clear();
			succ = false;
		}
	}

	if (succ)
	{
		mCycleRootDelta = CalcCycleRootDelta();
		Pose(mTime);
		mPose0 = GetPose();
		mVel0 = GetVel();
	}

	if (!succ)
	{
		printf("Failed to load motion from string.\n");
	}
	return succ;
}

const cMotion& cKinCharacter::GetMotion() const
{
	return mMotion;
}

double cKinCharacter::GetMotionDuration() const
{
	if (mMotion.IsValid())
	{
		return mMotion.GetDuration();
	}
	return 0;
}

void cKinCharacter::SetMotionDuration(double dur)
{
	return mMotion.SetDuration(dur);
}

void cKinCharacter::SetTime(double time)
{
	mTime = time;
}

double cKinCharacter::GetTime() const
{
	return mTime;
}

int cKinCharacter::GetCycle() const
{
	int cycle = 0;
	if (mMotion.EnableLoop())
	{
		double phase = mTime / mMotion.GetDuration();
		cycle = static_cast<int>(std::floor(phase));
	}
	return cycle;
}

double cKinCharacter::GetPhase() const
{
	double phase = mTime / mMotion.GetDuration();
	if (mMotion.EnableLoop())
	{
		phase -= static_cast<int>(phase);
		phase = (phase < 0) ? (1 + phase) : phase;
	}
	else
	{
		phase = cMathUtil::Clamp(phase, 0.0, 1.0);
	}
	return phase;
}

void cKinCharacter::Pose(double time)
{
	CalcPose(time, mPose);
	SetPose(mPose);

	CalcVel(time, mVel);
	SetVel(mVel);
}

void cKinCharacter::BuildAcc(Eigen::VectorXd& out_acc) const
{
	CalcAcc(mTime, out_acc);
}

bool cKinCharacter::HasMotion() const
{
	return mMotion.IsValid();
}

void cKinCharacter::SetRootPos(const tVector& pos)
{
	tVector root_pos = GetRootPos();
	tVector delta = pos - root_pos;
	MoveOrigin(delta);
}

void cKinCharacter::SetRootRotation(const tQuaternion& q)
{
	tQuaternion root_rot = GetRootRotation();
	tQuaternion dq = q * root_rot.conjugate();
	RotateOrigin(dq);
}

const tVector& cKinCharacter::GetOriginPos() const
{
	return mOrigin;
}

void cKinCharacter::SetOriginPos(const tVector& origin)
{
	tVector delta = origin - mOrigin;
	MoveOrigin(delta);
	mOrigin = origin; // this is needed in canse of NaNs
}

void cKinCharacter::MoveOrigin(const tVector& delta)
{
	mOrigin += delta;

	tVector root0 = cKinTree::GetRootPos(mJointMat, mPose0);
	root0 += delta;
	cKinTree::SetRootPos(mJointMat, root0, mPose0);

	tVector root = cKinTree::GetRootPos(mJointMat, mPose);
	root += delta;
	cKinTree::SetRootPos(mJointMat, root, mPose);
}

const tQuaternion& cKinCharacter::GetOriginRot() const
{
	return mOriginRot;
}

void cKinCharacter::SetOriginRot(const tQuaternion& rot)
{
	tQuaternion delta_rot = cMathUtil::QuatDiff(mOriginRot, rot);
	RotateOrigin(delta_rot);
	mOriginRot = rot; // this is needed in case of NaNs
}

void cKinCharacter::RotateOrigin(const tQuaternion& rot)
{
	mOriginRot = rot * mOriginRot;
	mOriginRot.normalize();

	tVector root_pos = GetRootPos();
	tVector root_pos_delta = mOrigin - root_pos;
	root_pos_delta = cMathUtil::QuatRotVec(rot, root_pos_delta);
	mOrigin = root_pos + root_pos_delta;

	tQuaternion root_rot0 = cKinTree::GetRootRot(mJointMat, mPose0);
	root_rot0 = rot * root_rot0;
	root_rot0.normalize();
	cKinTree::SetRootRot(mJointMat, root_rot0, mPose0);

	tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, mPose);
	root_rot = rot * root_rot;
	root_rot.normalize();
	cKinTree::SetRootRot(mJointMat, root_rot, mPose);

	tVector vel0 = cKinTree::GetRootVel(mJointMat, mVel0);
	vel0 = cMathUtil::QuatRotVec(rot, vel0);
	cKinTree::SetRootVel(mJointMat, vel0, mVel0);

	tVector vel = cKinTree::GetRootVel(mJointMat, mVel);
	vel = cMathUtil::QuatRotVec(rot, vel);
	cKinTree::SetRootVel(mJointMat, vel, mVel);

	tVector ang_vel0 = cKinTree::GetRootAngVel(mJointMat, mVel0);
	ang_vel0 = cMathUtil::QuatRotVec(rot, ang_vel0);
	cKinTree::SetRootAngVel(mJointMat, ang_vel0, mVel0);

	tVector ang_vel = cKinTree::GetRootAngVel(mJointMat, mVel);
	ang_vel = cMathUtil::QuatRotVec(rot, ang_vel);
	cKinTree::SetRootAngVel(mJointMat, ang_vel, mVel);
}

tVector cKinCharacter::GetRootAngVel() const
{	
	return cKinTree::GetRootAngVel(mJointMat, GetVel());
}

void cKinCharacter::ResetParams()
{ 
	cCharacter::ResetParams();
	mTime = 0;
}

tVector cKinCharacter::GetCycleRootDelta() const
{
	tVector delta = cMathUtil::QuatRotVec(mOriginRot, mCycleRootDelta);
	return delta;
}

tVector cKinCharacter::CalcCycleRootDelta() const
{
	int num_frames = mMotion.GetNumFrames();
	Eigen::VectorXd frame_beg = mMotion.GetFrame(0);
	Eigen::VectorXd  frame_end = mMotion.GetFrame(num_frames - 1);

	tVector root_pos_beg = cKinTree::GetRootPos(mJointMat, frame_beg);
	tVector root_pos_end = cKinTree::GetRootPos(mJointMat, frame_end);

	tVector delta = root_pos_end - root_pos_beg;
	return delta;
}

void cKinCharacter::CalcPose(double time, Eigen::VectorXd& out_pose) const
{
	tVector root_delta = tVector::Zero();
	tQuaternion root_delta_rot = tQuaternion::Identity();

	if (HasMotion())
	{
		mMotion.CalcFrame(time, out_pose);
		if (mMotion.EnableLoop())
		{
			int cycle_count = mMotion.CalcCycleCount(time);
			root_delta = cycle_count * mCycleRootDelta;
		}
	}
	else
	{
		out_pose = mPose0;
	}

	tVector root_pos = cKinTree::GetRootPos(mJointMat, out_pose);
	//printf("root_pos:%f\t%f\t%f\n", root_pos[0], root_pos[1], root_pos[2]);sleep(0.1);
	tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, out_pose);

	root_delta_rot = mOriginRot * root_delta_rot;
	root_rot = root_delta_rot * root_rot;
	root_pos += root_delta;
	root_pos = cMathUtil::QuatRotVec(root_delta_rot, root_pos);
	root_pos += mOrigin;

	cKinTree::SetRootPos(mJointMat, root_pos, out_pose);
	cKinTree::SetRootRot(mJointMat, root_rot, out_pose);
}

void cKinCharacter::CalcVel(double time, Eigen::VectorXd& out_vel) const
{
	if (HasMotion())
	{
		mMotion.CalcFrameVel(time, out_vel);
	}
	else
	{
		out_vel = Eigen::VectorXd::Zero(GetNumDof());
	}
}

void cKinCharacter::CalcAcc(double time, Eigen::VectorXd& out_acc) const
{
	Eigen::VectorXd vel0;
	Eigen::VectorXd vel1;
	CalcVel(time - gDiffTimeStep, vel0);
	CalcVel(time, vel1);
	out_acc = (vel1 - vel0) / gDiffTimeStep;
}

bool cKinCharacter::IsMotionOver() const
{
	bool over = true;
	if (HasMotion())
	{
		over = mMotion.IsOver(mTime);
	}

	return over;
}

void cKinCharacter::BlendFrames(const cMotion::tFrame* a, const cMotion::tFrame* b, double lerp, cMotion::tFrame* out_frame) const
{
	cKinTree::LerpPoses(mJointMat, *a, *b, lerp, *out_frame);
}

void cKinCharacter::MirrorFrame(const std::vector<int>* right_joints, const std::vector<int>* left_joints, cMotion::tFrame* out_frame) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::MirrorPoseStance(joint_mat, *left_joints, *right_joints, *out_frame);
}

void cKinCharacter::CalcFrameVel(const cMotion::tFrame* a, const cMotion::tFrame* b, double timestep, cMotion::tFrame* out_vel) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::CalcVel(joint_mat, *a, *b, timestep, *out_vel);
}

void cKinCharacter::PostProcessFrame(cMotion::tFrame* out_frame) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::PostProcessPose(joint_mat, *out_frame);
}

std::vector<double> cKinCharacter::CalcKeyframeTime(std::vector<int> IKjoints)
{
	std::vector<double> keyframe_time;
	int joint1;
	int joint2;
	if(IKjoints.size() == 1){
		joint1 = IKjoints[0];
		joint2 = 0;
	} else if (IKjoints.size() == 2){
		joint1 = IKjoints[0];
		joint2 = IKjoints[1];
	} else {
		assert(false);
	}
		
	double dur = GetMotionDuration();
	double frame_time = 1.0 / 30.0;
	int checkpoint_num = floor(dur / frame_time);

	std::vector<float> dist;
	for (int i = 0; i < checkpoint_num; i++) {
		double time = i * frame_time;
		cKinCharacter::SetTime(time);
		cKinCharacter::Pose(time);
		tVector vec = cKinCharacter::CalcJointPos(joint1) - cKinCharacter::CalcJointPos(joint2);
		vec[1] = 0.0; vec[3] = 0.0;
		dist.push_back(float(vec.norm()));
	}
	std::vector<int> idx;
	idx.push_back(std::distance(dist.begin(), std::max_element(dist.begin(), dist.end())));
	for (int i = 0; i < idx.size(); i++){
		double time = idx[i] * frame_time;
		keyframe_time.push_back(time);
	}

	return keyframe_time;
}

std::vector<double> cKinCharacter::CalcPertitionTime(std::vector<int> IKjoints, std::vector<double> keyframe_time)
{
	std::vector<double> pertition_time;
	int joint1;
	int joint2;
	if(IKjoints.size() == 1){
		joint1 = IKjoints[0];
		joint2 = 0;
	} else if (IKjoints.size() == 2){
		joint1 = IKjoints[0];
		joint2 = IKjoints[1];
	} else {
		assert(false);
	}

	double dur = GetMotionDuration();
	for (int i = 0; i < keyframe_time.size(); i++){
		if ( i + 1 == keyframe_time.size()){
			pertition_time.push_back(dur);
		} else {
			double frame_time = 1.0 / 30.0;
			double interval = keyframe_time[i + 1] - keyframe_time[i];
			int checkpoint_num = floor(interval / frame_time);
			std::vector<double> dist;
			for (int j = 0; j < checkpoint_num; j ++){
				double time = keyframe_time[i] + j * frame_time;
				cKinCharacter::SetTime(time);
				cKinCharacter::Pose(time);
				tVector vec = cKinCharacter::CalcJointPos(joint1) - cKinCharacter::CalcJointPos(joint2);
				vec[1] = 0.0; vec[3] = 0.0;
				dist.push_back(float(vec.norm()));
			}
			int argMin = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
			pertition_time.push_back(keyframe_time[i] + argMin * frame_time);
		}
	}

	return pertition_time;

}

std::vector<int> cKinCharacter::ChooseIKjoints(std::vector<int> IKjoints, std::vector<double> keyframe_time)
{
	std::vector<int> IKjoint_per_KF;
	for (int i = 0; i < keyframe_time.size(); i++){
		if (IKjoints.size() == 1){
			IKjoint_per_KF.push_back(IKjoints[0]);
		} else {
			double time = keyframe_time[i];
			cKinCharacter::SetTime(time);
			cKinCharacter::Pose(time);
			std::vector<double> dist;
			for (int j = 0; j < IKjoints.size(); j++){
				tVector vec1 = cKinCharacter::CalcJointPos(IKjoints[j]) - cKinCharacter::CalcJointPos(0); vec1[1] = 0.0; vec1[3] = 0.0;
				tVector vec2 = cKinCharacter::CalcJointVel(0); vec2[1] = 0.0; vec2[3] = 0.0;
				dist.push_back(vec1.dot(vec2));
			}
			int argMax = std::distance(dist.begin(), std::max_element(dist.begin(), dist.end()));
			IKjoint_per_KF.push_back(IKjoints[argMax]);
		}
	}

	return IKjoint_per_KF;
}

