#pragma once

#include "SceneAug.h"
#include "sim/RBDUtil.h"
#include "sim/CtController.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "SceneImitate.h"

#include <Eigen/Dense>

#include <numeric>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

const int gValLogSize = 1000 * 4;

double cSceneAug::CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{	
	double pose_w = 0.5;
	double vel_w = 0.2;
	double end_eff_w = 0.2;
	double root_w = 0.0;
	double com_w = 0.1;
	

	double style_w = 0.0;

	double total_w = pose_w + vel_w + end_eff_w + root_w + com_w + style_w;
	pose_w /= total_w;
	vel_w /= total_w;
	end_eff_w /= total_w;
	root_w /= total_w;
	com_w /= total_w;
	style_w /= total_w;

	const double pose_scale = 2;
	const double vel_scale = 0.005;
	const double end_eff_scale = 5;
	//const double root_scale = 5;
	const double root_scale = 0.0;
	const double com_scale = 100;

	const double style_scale = 0.1;
	
	const auto& joint_mat = sim_char.GetJointMat();
	const auto& body_defs = sim_char.GetBodyDefs();
	double reward = 0;

	
	const Eigen::VectorXd& pose0 = sim_char.GetPose();
	const Eigen::VectorXd& vel0 = sim_char.GetVel();
	////
	//const Eigen::VectorXd& pose0 = kin_char.GetPose();
	//const Eigen::VectorXd& vel0 = kin_char.GetVel();
	const Eigen::VectorXd& pose1 = kin_char.GetPose();
	const Eigen::VectorXd& vel1 = kin_char.GetVel();

	//std::cerr << vel0[0] << vel0[1] << vel0[2] << std::endl;
	//std::cerr << cKinTree::GetRootVel(joint_mat, vel1) << std::endl;

	///print vel
	///const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t", " ", "", "", "", "");std::cout << vel1 << std::endl;sleep(0.1);
	tMatrix origin_trans = sim_char.BuildOriginTrans();
	tMatrix kin_origin_trans = kin_char.BuildOriginTrans();
	////
	//origin_trans = kin_origin_trans;

	tVector com0_world = sim_char.CalcCOM();
	tVector com_vel0_world = sim_char.CalcCOMVel();
	tVector com1_world;
	tVector com_vel1_world;
	cRBDUtil::CalcCoM(joint_mat, body_defs, pose1, vel1, com1_world, com_vel1_world);
	////
	//com_vel0_world = com_vel1_world;

	int root_id = sim_char.GetRootID();
	tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0); //world pos
	tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
	tQuaternion root_rot0 = cKinTree::GetRootRot(joint_mat, pose0); //world rot
	tQuaternion root_rot1 = cKinTree::GetRootRot(joint_mat, pose1);
	tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
	tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
	tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
	tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);

	double pose_err = 0;
	double vel_err = 0;
	double end_eff_err = 0;
	double root_err = 0;
	double com_err = 0;
	double heading_err = 0;

	int num_end_effs = 0;
	int num_joints = sim_char.GetNumJoints();
	assert(num_joints == mJointWeights.size());

	double root_rot_w = mJointWeights[root_id];
	pose_err += root_rot_w * cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
	vel_err += root_rot_w * cKinTree::CalcRootAngVelErr(joint_mat, vel0, vel1);

	
	for (int j = root_id + 1; j < num_joints; ++j)
	{
		double w = mJointWeights[j];
		double curr_pose_err = cKinTree::CalcPoseErr(joint_mat, j, pose0, pose1);
		double curr_vel_err = cKinTree::CalcVelErr(joint_mat, j, vel0, vel1);
		pose_err += w * curr_pose_err;
		vel_err += w * curr_vel_err;

		bool is_end_eff = sim_char.IsEndEffector(j);
		if (is_end_eff)
		{
			tVector pos0 = sim_char.CalcJointPos(j);
			tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, j);
			////
			//pos0 = pos1;

			double ground_h0 = mGround->SampleHeight(pos0);
			double ground_h1 = kin_char.GetOriginPos()[1];
			////
			//ground_h0 = ground_h1;

			tVector pos_rel0 = pos0 - root_pos0;
			tVector pos_rel1 = pos1 - root_pos1;
			pos_rel0[1] = pos0[1] - ground_h0;
			pos_rel1[1] = pos1[1] - ground_h1;

			pos_rel0 = origin_trans * pos_rel0;
			pos_rel1 = kin_origin_trans * pos_rel1;

			double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
			end_eff_err += curr_end_err;
			++num_end_effs;
		}
	}

	if (num_end_effs > 0)
	{
		end_eff_err /= num_end_effs;
	}
	

	double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos());
	double root_ground_h1 = kin_char.GetOriginPos()[1];
	////
	//root_ground_h0 = root_ground_h1;
	root_pos0[1] -= root_ground_h0;
	root_pos1[1] -= root_ground_h1;
	double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
	
	double root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1);
	root_rot_err *= root_rot_err;

	double root_vel_err = (root_vel1 - root_vel0).squaredNorm();
	double root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm();

	
	root_err = root_pos_err
			+ 0.1 * root_rot_err
			+ 0.01 * root_vel_err
			+ 0.001 * root_ang_vel_err;
	
	root_err = root_rot_err + 0.01 * root_ang_vel_err;
	com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm();

	double pose_reward = exp(-pose_scale * pose_err);
	double vel_reward = exp(-vel_scale * vel_err);
	double end_eff_reward = exp(-end_eff_scale * end_eff_err);
	double root_reward = exp(-root_scale * root_err);
	double com_reward = exp(-com_scale * com_err);

		
	reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward
		+ root_w * root_reward + com_w * com_reward;
	

	//reward = pose_reward * vel_reward * end_eff_reward * root_reward * com_reward;
	
	//double style_reward = exp(-err_scale * style_scale * style_err);

	//reward = style_w * style_reward + root_w * root_reward;
	return reward;
}

void cSceneAug::CalcReward(const cSimCharacter& sim_char, const cKinCharacter& ref_char, double &r, double &r_i, double &r_s) const
{
	double omega_goal = 0.3 * mEnableStrikeReward;
	double omega_imitate = 0.7;
	double omega_total = omega_goal + omega_imitate;
	omega_goal /= omega_total;
	omega_imitate /= omega_total;

	double r_imitate = CalcRewardImitate(sim_char, ref_char);
	double r_goal = CalcRewardGoal(sim_char, ref_char);

	double reward = omega_imitate * r_imitate + omega_goal * r_goal;
	//printf("%lf\n%lf\n%lf\n\n\n\n", reward, r_imitate , r_goal);sleep(0.1);

	/*
	std::ofstream rrec("rrec.txt", std::ios::app);
	if(rrec)
	{
		rrec << std::fixed << r_imitate << "\t" << r_goal << "\t" << reward << "\n";
		//rrec << std::fixed << r_goal << "\t";
		//rrec << std::fixed << reward << "\n";
	}
	rrec.close();
	*/
	
	r = reward; r_i = r_imitate; r_s = r_goal;
}

double cSceneAug::CalcRewardGoal(const cSimCharacter& sim_char, const cKinCharacter& ref_char) const
{
	auto ctrl = dynamic_cast<cCtController*>(sim_char.GetController().get());
	cSimGoal goal = ctrl->GetGoal(ctrl->GetTime());
	int idx = ctrl->ChoosePertitionIdx();
	int ikjoint = mIKJoint[idx];
	tVector pos_goal = goal.GetPos();
	int flag_goal_cleared = goal.GetFlagClear();

	double scale_goal = 4;

	tVector pos_ef = sim_char.GetBodyPart(ikjoint)->GetPos();
	pos_ef = tVector(pos_ef[0], pos_ef[1], pos_ef[2], 0);
	double GoalTime = pos_goal[3];
	pos_goal = tVector(pos_goal[0], pos_goal[1], pos_goal[2], 0);
	double err_goal = (pos_ef - pos_goal).squaredNorm();
 	const auto& kin_char = GetKinChar();
	double mTime = kin_char->GetTime();
	double time_factor = std::max(0.0, 1 - std::abs((GoalTime - mTime)/GoalTime));
	time_factor = std::pow(time_factor, mTimeFactorPow);
	
	double r_goal = 0;
	if(mRewardType == 0){	
		r_goal = exp(-1 * mScaleGoalReward * err_goal);
	}else if (mRewardType == 1){
		r_goal = exp(-1 * mScaleGoalReward * err_goal * time_factor);
	}else if (mRewardType == 2){
		r_goal = exp(-1 * mScaleGoalReward * err_goal) * time_factor;
	}else if (mRewardType == 3){
		if(flag_goal_cleared){
			r_goal = mRewardBuffer;
		}else{
			r_goal = exp(-1 * mScaleGoalReward * err_goal) * time_factor;
		}
	}else if (mRewardType == 4){
		r_goal = exp(-1 * mScaleGoalReward * err_goal / (time_factor + 0.01));
	}else if (mRewardType == 5){
		///hmm...
		///How do you let agents know about information of C(x,y,z,t) within one episode? 
	}else{
		r_goal = 0;
	}
	/*
	printf("%f %f %f\n%f %f %f %f\n\n",
	       pos_goal[0], pos_goal[1], pos_goal[2],
	       pos_ef[0], pos_ef[1], pos_ef[2],
	       r_goal);
	*/
	//printf("%f\n", mTime);
	
	return r_goal;
}


cSceneAug::cSceneAug()
{
	mEnableRandRotReset = false;
	mSyncCharRootPos = true; ////
	mSyncCharRootRot = false;  ////
	mMotionFile = "";
	mMotionString = "";
	mEnableRootRotFail = false;
	mHoldEndFrame = 0;

	mRewardType = 0;

	mRewardLog.Reserve(gValLogSize);
	mIRewardLog.Reserve(gValLogSize);
	mSRewardLog.Reserve(gValLogSize);
}

cSceneAug::~cSceneAug()
{
}

void cSceneAug::ParseArgs(const std::shared_ptr<cArgParser>& parser)
{
	cRLSceneSimChar::ParseArgs(parser);
	parser->ParseString("motion_file", mMotionFile);
	
	parser->ParseBool("enable_rand_rot_reset", mEnableRandRotReset);
	parser->ParseBool("sync_char_root_pos", mSyncCharRootPos);
	parser->ParseBool("sync_char_root_rot", mSyncCharRootRot);
	parser->ParseBool("enable_root_rot_fail", mEnableRootRotFail);
	parser->ParseDouble("hold_end_frame", mHoldEndFrame);
	parser->ParseBool("early_termination", mEarlyTermination);

	parser->ParseBool("enable_strike_reward", mEnableStrikeReward);
	
	parser->ParseString("mode", mAugMode);
	std::cerr << "aug mode: " << mAugMode << std::endl;
	if (mAugMode == "IK"){
		//parser->ParseString("goal_shape", mGoalShape);
		parser->ParseString("sample_shape", mGoalShape);
		parser->ParseInts("end_effector", mIKJoint);
		parser->ParseInts("keyframe", mKeyframe);
		parser->ParseDoubles("sample_param_plus", mSampleParamPlus);
		parser->ParseDoubles("sample_param_minus", mSampleParamMinus);
		parser->ParseDouble("windowtime", mWindowTime);
	} else if (mAugMode == "VAE"){
		mSample = true;
		parser->ParseString("class", mClass);
		parser->ParseStrings("subject", mSubject);
		parser->ParseString("sampler_arg_file", mSamplerArgFile);
		mEnableStrikeReward = false;
	} else if (mAugMode == "imitate"){
		
	}else {
		exit(0);
	}
	
	parser->ParseInt("reward_type", mRewardType);
	parser->ParseDouble("time_factor_pow", mTimeFactorPow);	
	parser->ParseDouble("scale_goal_reward", mScaleGoalReward);

	parser->ParseBool("start_from_beginning", mStartFromBeginning);
}

void cSceneAug::Init()
{
	InitPython();
	InitJava();
	mKinChar.reset();
	cRLSceneSimChar::Init();
	///BuildKinChar();
	InitJointWeights();
	ResetPoseLog();
}

int cSceneAug::PyInit()
{
	Py_Initialize();
	import_array();
}

void cSceneAug::InitPython()
{
	PyInit();
	PyObject *sys = PyImport_ImportModule("sys");
	PyObject *path = PyObject_GetAttrString(sys, "path");
	PyList_Append(path, PyUnicode_DecodeFSDefault("."));

	PyObject *warnings = PyImport_ImportModule("warnings");
	PyObject *pMName = PyUnicode_FromString("simplefilter");
	PyObject *pArg   = PyTuple_New(1);
	PyTuple_SetItem(pArg, 0, PyUnicode_FromString("ignore"));
	PyObject_CallMethodObjArgs(warnings, pMName, PyUnicode_FromString("ignore"), NULL);
	Py_DECREF(pMName);
	Py_DECREF(pArg);

	
	if (mAugMode == "VAE"){
		InitSampler();
	}

}

void cSceneAug::InitSampler()
{
	PyObject *pName = PyUnicode_DecodeFSDefault("motionAE.src.sample");
	PyObject *pModule = PyImport_Import(pName);
	if (pModule == nullptr) {
		PyErr_Print();
		std::cerr << "Fails to import modules. \n";
		return;
	}
	Py_DECREF(pName);

	PyObject *dict = PyModule_GetDict(pModule);
	if (dict == nullptr) {
		PyErr_Print();
		std::cerr << "Fails to get the dictionary. \n";
		return;
	}
	Py_DECREF(pModule);

	PyObject *python_class = PyDict_GetItemString(dict, "Sampler");
	if (python_class == nullptr) {
		PyErr_Print();
		std::cerr << "Fails to get the Python class.\n";
		return;
	}
	Py_DECREF(dict);
	
	if (PyCallable_Check(python_class)) {
		int offset = 5;
		PyObject *pList = PyList_New(offset + mSubject.size());
		PyObject *pArgs = PyTuple_New(1);
		PyList_SetItem(pList, 0, PyUnicode_FromString("--arg_file"));
		PyList_SetItem(pList, 1, PyUnicode_FromString(mSamplerArgFile.c_str()));
		PyList_SetItem(pList, 2, PyUnicode_FromString("--class"));
		PyList_SetItem(pList, 3, PyUnicode_FromString(mClass.c_str()));
		PyList_SetItem(pList, 4, PyUnicode_FromString("--subject"));
		for (int i=0;i<mSubject.size();i++){
			PyList_SetItem(pList, offset + i, PyUnicode_FromString(mSubject[i].c_str()));
		}
		PyTuple_SetItem(pArgs, 0, pList);
    	mSampler = PyObject_CallObject(python_class, pArgs);
		Py_DECREF(pList);
		Py_DECREF(pArgs);
  	} else {
    	std::cout << "Cannot instantiate the Python class" << std::endl;
    return;
  	}
	Py_DECREF(python_class);
}

void cSceneAug::InitJava()
{
	// JavaVM起動オプションの設定
	JavaVMOption options[1];
    //options[0].optionString = "-Djava.class.path=./DeepMimicCore/anim/";
	options[0].optionString = "-Djava.class.path="
	":./lib/jackson-core-2.9.7.jar"
	":./lib/jackson-databind-2.9.7.jar"
	":./lib/jackson-annotations-2.9.7.jar"
	":./lib/vecmath.jar"
	":./lib/caliko/caliko-demo/target/caliko-demo-1.3.7-jar-with-dependencies.jar";
	
	
 
    JavaVMInitArgs vm_args;
    vm_args.version = JNI_VERSION_10;
    vm_args.options = options;
    vm_args.nOptions = 1;

	JavaVM *jvm;
	JNIEnv *env;

	// JavaVMを初期化，起動
	int res = JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);
    if(res){
        std::cout << "cannot run JavaVM : " << res << std::endl;
		std::exit(1);
		return;
    }
	
	res = jvm->DetachCurrentThread();
	if (res < 0){
		std::cerr << "Java VM detach faild:" << res << std::endl;
		return;
	}
	
}


///Set Goal in every new episode
void cSceneAug::Reset()
{	
	//Init();
	/*
    const auto& kin_char = GetKinChar();
	const auto& ctrl = dynamic_cast<cCtController*>(GetCharacter()->GetController().get());
    if (StartedNearBeginning() && kin_char->IsMotionOver() && CheckRewardLog())
    {
        ctrl->UpdateGoalEllipsoid();
    }
	*/
	ResetScene();

	mRewardBuffer = 0;
	mRewardBuffer_ = 0;

	ResetPoseLog();

	mRewardLog.Clear();
	mIRewardLog.Clear();
	mSRewardLog.Clear();
}

bool cSceneAug::CheckRewardLog()
{
	int num_val = static_cast<int>(mIRewardLog.GetSize());
  	double sum_r_imitate = 0;
	for (int i=0; i < num_val; ++i)
	{
		sum_r_imitate += mIRewardLog[i];
	}
	double mean_r_imitate = sum_r_imitate / num_val;
	double last_r_imitate = mIRewardLog[num_val-1];
  	const auto& kin_char = GetKinChar();

  	//printf("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", mDeltaDminus, mDeltaDplus, mDeltaHminus, mDeltaHplus, mDeltaYminus, mDeltaYplus, mDeltaTminus, mDeltaTplus);

	/*
	printf("mean_r_imitate %f r_strike %f %f %d\t %d %d %d %d %d \n", 
	mean_r_imitate, mRewardBuffer_, kin_char->GetMotionDuration(), mIRewardLog.GetSize(), mIRewardLog.GetCapacity(), 
	kin_char->IsMotionOver(), mRewardBuffer_ > 0.7, mean_r_imitate > 0.7, (mRewardBuffer_ > 0.7) && (mean_r_imitate > 0.7));
	*/
  	return (mRewardBuffer_ > 0.81) && (mean_r_imitate > 0.7) && (last_r_imitate > 0.5);
}

bool cSceneAug::StartedNearBeginning()
{
	const auto& kin_char = GetKinChar();
	double GoalTime = GetGoal().GetPos()[3];
	double ratio = mInitTime / GoalTime;
	//printf("initTime %f GoalTime %f ratio %f startedNearBeginning %d\n", mInitTime, GoalTime, ratio, ratio < 0.1);sleep(0.1);
	return (ratio < 0.1);
}

void cSceneAug::ResetPoseLog() 
{
	const cSimCharacter *sim_char = GetAgentChar(0);
	Eigen::Matrix<double, 1, 60> sim_pose = GetSimAbsPose(*sim_char);
	Eigen::Matrix<double, 1, 60> kin_pose = GetKinAbsPose(*sim_char, *mKinChar);	

	int rows = mSimPoseLog.rows();
	int cols = mSimPoseLog.cols();

	for (int i = 0; i < rows; ++i)
	{
		mSimPoseLog.block(i,0,1,cols) = sim_pose;
		mKinPoseLog.block(i,0,1,cols) = kin_pose;
	}
}

PyObject* cSceneAug::Matrix2PyList(const Eigen::Matrix<double, 16, 60> mat) const
{
	int rows = mat.rows();
	int cols = mat.cols();

	PyObject *listObj = PyList_New(rows);
	Eigen::MatrixXd slice;
	std::vector<double> vec;
	for (int i = 0; i < rows; i++)
	{
		slice = mat.block(i,0,1,cols);
		std::vector<double> vec(slice.data(), slice.data() + slice.size());
		PyList_SET_ITEM(listObj, i, std_vector_to_py_list(vec));
	}
	return listObj;
}

PyObject* cSceneAug::std_vector_to_py_list(const std::vector<double>& v) const
{
	PyObject *listObj = PyList_New( v.size() );
	for (int i = 0; i < v.size(); i++) 
	{	
		PyObject *num = PyFloat_FromDouble( (double) v[i]);
		if (!num) 
		{
			Py_DECREF(listObj);
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}

void cSceneAug::UpdateLog()
{
	const cSimCharacter *sim_char = GetAgentChar(0);
	Eigen::Matrix<double, 1, 60> sim_pose = GetSimAbsPose(*sim_char);
	Eigen::Matrix<double, 1, 60> kin_pose = GetKinAbsPose(*sim_char, *mKinChar);	
	
	int cols = mSimPoseLog.cols();
	int rows = mSimPoseLog.rows();
	mSimPoseLog.block(0,0,rows-1, cols) = mSimPoseLog.block(1,0,rows-1,cols);
	mKinPoseLog.block(0,0,rows-1, cols) = mKinPoseLog.block(1,0,rows-1,cols);

	mSimPoseLog.block(rows-1, 0, 1, cols) = sim_pose;
	mKinPoseLog.block(rows-1, 0, 1, cols) = kin_pose;
	
}

Eigen::VectorXd cSceneAug::GetSimAbsPose(const cSimCharacter& sim_char) const
{
	//Return Abs Pose to ROOT joint
	tMatrix origin_trans = sim_char.BuildOriginTrans();
	tQuaternion origin_quat = cMathUtil::RotMatToQuaternion(origin_trans);

	int num_parts = sim_char.GetNumBodyParts();
	int root_id = sim_char.GetRootID();

	int rot_dim = 4;

	Eigen::VectorXd out_pose(rot_dim * (num_parts - 1));

	int idx = 0;
	for (int i = 0; i < num_parts; ++i)
	{
		if (sim_char.IsValidBodyPart(i))
		{
			//const auto& curr_part = sim_char.GetBodyPart(i);			
			//tQuaternion curr_quat = curr_part->GetRotation();
			tQuaternion curr_quat = sim_char.CalcJointWorldRotation(i);
			if (i == root_id)
			{
				continue;
			}
			else
			{
				curr_quat = origin_quat.inverse() * curr_quat;
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
	return out_pose;
}

Eigen::VectorXd cSceneAug::GetKinAbsPose(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{
	//return abs pose to ROOT joint
	tMatrix origin_trans = kin_char.BuildOriginTrans();
	tQuaternion origin_quat = cMathUtil::RotMatToQuaternion(origin_trans);
	
	int num_parts = sim_char.GetNumBodyParts();
	int root_id = sim_char.GetRootID();

	int rot_dim = 4;

	Eigen::VectorXd out_pose(rot_dim * (num_parts - 1));

	int idx = 0;
	for (int i = 0; i < num_parts; ++i)
	{
		if (sim_char.IsValidBodyPart(i))
		{
			tQuaternion curr_quat = kin_char.CalcJointWorldRotation(i);
			if (i == root_id)
			{
				continue;
			}
			else
			{
				curr_quat = origin_quat.inverse() * curr_quat;
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
	return out_pose;
}

double cSceneAug::CalcReward(int agent_id) const
{
	const cSimCharacter* sim_char = GetAgentChar(agent_id);
	bool fallen = HasFallen(*sim_char);

	double r = 0;
	int max_id = 0;
	if (!fallen)
	{
		double _;
		CalcReward(*sim_char, *mKinChar, r, _, _);
	}
	return r;
}

const std::shared_ptr<cKinCharacter>& cSceneAug::GetKinChar() const
{
	return mKinChar;
}

void cSceneAug::EnableRandRotReset(bool enable)
{
	mEnableRandRotReset = enable;
}

bool cSceneAug::EnabledRandRotReset() const
{
	bool enable = mEnableRandRotReset;
	return enable;
}

cSceneAug::eTerminate cSceneAug::CheckTerminate(int agent_id) const
{
	eTerminate terminated = (mEarlyTermination) ? cRLSceneSimChar::CheckTerminate(agent_id) : eTerminateNull;
	if (terminated == eTerminateNull)
	{
		bool end_motion = false;
		const auto& kin_char = GetKinChar();
		const cMotion& motion = kin_char->GetMotion();

		if (motion.GetLoop() == cMotion::eLoopNone)
		{
			double dur = motion.GetDuration();
			double kin_time = kin_char->GetTime();
			end_motion = kin_time > dur + GetHoldEndFrame();
			//if(end_motion){printf("End motion\nkin_time: %f\nduration: %f\n", kin_time, dur);sleep(float(0.1));}
		}
		else
		{
			end_motion = kin_char->IsMotionOver();
		}

		terminated = (end_motion) ? eTerminateFail : terminated;
	}
	
	return terminated;
}

std::string cSceneAug::GetName() const
{
	return "Strike";
}

bool cSceneAug::BuildCharacters()
{
	bool succ = cRLSceneSimChar::BuildCharacters();
	if (EnableSyncChar())
	{
		SyncCharacters();
	}
	return succ;
}

void cSceneAug::CalcJointWeights(const std::shared_ptr<cSimCharacter>& character, Eigen::VectorXd& out_weights) const
{
	int num_joints = character->GetNumJoints();
	out_weights = Eigen::VectorXd::Ones(num_joints);
	for (int j = 0; j < num_joints; ++j)
	{
		double curr_w = character->GetJointDiffWeight(j);
		out_weights[j] = curr_w;
	}

	double sum = out_weights.lpNorm<1>();
	out_weights /= sum;
}


bool cSceneAug::BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCtController>& out_ctrl)
{
	bool succ = cSceneSimChar::BuildController(ctrl_params, out_ctrl);
	if (succ)
	{
	    BuildKinChar();
		const auto& kin_char = GetKinChar();
		tVector root_vec;
		if (mIKJoint.size() > 0){
			if (mKeyframe.size() == 0){
				mCenterT = kin_char->CalcKeyframeTime(mIKJoint);
			} else {
				mCenterT = {};
				for (int i = 0; i < mKeyframe.size(); i++){
					mCenterT.push_back(kin_char->GetMotion().GetFrameTime(mKeyframe[i]));
				}
			}
			mPertitionTime = kin_char->CalcPertitionTime(mIKJoint, mCenterT);
			mIKJoint = kin_char->ChooseIKjoints(mIKJoint, mCenterT);
			std::vector<tVector> root_vec_arr = {};
			for (int i = 0; i < mCenterT.size(); i++){
				kin_char->SetTime(mCenterT[i]);
				kin_char->Pose(mCenterT[i]);
				int ikj = mIKJoint[i];
				tVector center = kin_char->CalcJointPos(ikj);
				root_vec = kin_char->CalcJointPos(ikj - 2);
				mCenter.push_back(tVector(center[0], center[1], center[2], mCenterT[i]));
				root_vec_arr.push_back(root_vec);
			}
			out_ctrl->SetGoalCenter(mCenter);
			out_ctrl->SetPertitionTime(mPertitionTime);
			out_ctrl->SetRootVec(root_vec_arr);
			//std::ofstream centerfile("center_from_root.txt", std::ios::app);
			//centerfile << mCenterD - root_vec[0] << " " << mCenterH - root_vec[1] << " " << mCenterY - root_vec[2] << std::endl;

			//exit(0);
		}
		
		out_ctrl->SetGoalShape(mGoalShape);
		if (mGoalShape == "Ellipsoid")
		{
			out_ctrl->SetGoalEllipsoidMinus(tVector(mSampleParamMinus.data()));
			out_ctrl->SetGoalEllipsoidPlus(tVector(mSampleParamPlus.data()));
		} else if(mGoalShape == "FanShape")
		{
			out_ctrl->SetGoalFanShapeHigh(tVector(mSampleParamPlus.data()));
			out_ctrl->SetGoalFanShapeLow(tVector(mSampleParamMinus.data()));
		} else if(mGoalShape == "Fixed"){
		}
		
		
		if (mIKJoint.size() > 0)
		{
			out_ctrl->SampleGoalPos();
			std::vector<cSimGoal> goal = out_ctrl->GetGoal();
			std::vector<tVector> goal_pos = {};
			for (int i = 0; i < goal.size(); i++){
				goal_pos.push_back(goal[i].GetPos());
			}
			//std::cerr << goal[0].GetPos() << std::endl;
			BuildKinChar(goal_pos);
		}else if (mSample) {
			BuildKinCharSample();
		}
		
		auto ct_ctrl = out_ctrl;
		if (ct_ctrl != nullptr)
		{	
			const auto& kin_char = GetKinChar();
			double cycle_dur = kin_char->GetMotionDuration();
			ct_ctrl->SetCyclePeriod(cycle_dur);
			ct_ctrl->SetKinChar(kin_char);
		}
		
		
	}
	return succ;
}

void cSceneAug::BuildKinChar()
{
	bool succ = BuildKinCharacter(0, mKinChar);
	if (!succ)
	{
	        printf("Failed to build kin character\n");
		assert(false);
	}
}

void cSceneAug::BuildKinChar(std::vector<tVector> GoalPos)
{
	bool succ = BuildKinCharacter(0, mKinChar, GoalPos);
	if (!succ)
	{
		printf("Failed to build kin character\n");
		assert(false);
	}
}

void cSceneAug::BuildKinCharSample()
{
	//generate motion file
	PyObject* pName = PyUnicode_FromString("sample_MimicMotion");
	//std::cerr << mMotionFile << std::endl;
	PyObject* result;
	char* name;
	if(PyObject_HasAttr(mSampler, pName)){
		result = PyObject_CallMethodObjArgs(mSampler, pName, PyUnicode_FromString(mMotionFile.c_str()), NULL);
	}
	name = PyUnicode_AsUTF8(result);
	std::string str(name);
	mMotionString = name;
	Py_DECREF(pName);
	Py_DECREF(result);
	BuildKinChar();
}

bool cSceneAug::BuildKinCharacter(int id, std::shared_ptr<cKinCharacter>& out_char) const
{
	auto kin_char = std::shared_ptr<cKinCharacter>(new cKinCharacter());
	const cSimCharacter::tParams& sim_char_params = mCharParams[0];
	cKinCharacter::tParams kin_char_params;

	kin_char_params.mID = id;
	kin_char_params.mCharFile = sim_char_params.mCharFile;
	kin_char_params.mOrigin = sim_char_params.mInitPos;
	kin_char_params.mLoadDrawShapes = false;
	kin_char_params.mMotionFile = mMotionFile;
	kin_char_params.mMotionString = mMotionString;
	
	kin_char_params.mGoalPos = {};
	kin_char_params.mCenter = {};
	kin_char_params.mIKJoint = {};
	kin_char_params.mPertitionTime = {};
	kin_char_params.mWindowTime = 0.0;

	bool succ = kin_char->Init(kin_char_params);
	if (succ)
	{
		out_char = kin_char;
	}
	
	return succ;
}

bool cSceneAug::BuildKinCharacter(int id, std::shared_ptr<cKinCharacter>& out_char, std::vector<tVector> GoalPos) const
{
	auto kin_char = std::shared_ptr<cKinCharacter>(new cKinCharacter());
	const cSimCharacter::tParams& sim_char_params = mCharParams[0];
	cKinCharacter::tParams kin_char_params;

	kin_char_params.mID = id;
	kin_char_params.mCharFile = sim_char_params.mCharFile;
	kin_char_params.mOrigin = sim_char_params.mInitPos;
	kin_char_params.mLoadDrawShapes = false;
	kin_char_params.mMotionFile = mMotionFile;
	
	kin_char_params.mGoalPos = GoalPos;
	kin_char_params.mCenter = mCenter;
	kin_char_params.mIKJoint = mIKJoint;
	kin_char_params.mPertitionTime = mPertitionTime;
	kin_char_params.mWindowTime = mWindowTime;
	
	bool succ = kin_char->Init(kin_char_params);
	if (succ)
	{
		out_char = kin_char;
	}
	
	return succ;
}


void cSceneAug::UpdateCharacters(double timestep)
{
	UpdateKinChar(timestep);
	cRLSceneSimChar::UpdateCharacters(timestep);

	const cSimCharacter* sim_char = GetAgentChar(0);
	bool fallen = HasFallen(*sim_char);

	double reward, r_imitate, r_strike;
	UpdateLog();
	CalcReward(*sim_char, *mKinChar, reward, r_imitate, r_strike);
	mRewardBuffer = r_strike;

	if (mCenterT.size() > 0){
		CheckGoalCleared();
		auto ctrl = dynamic_cast<cCtController*>(sim_char->GetController().get());
		cSimGoal goal = ctrl->GetGoal(ctrl->GetTime());
		tVector goal_pos = goal.GetPos();
		double goal_time = goal_pos[3];
		const auto& kin_char = GetKinChar();
		double time = kin_char->GetTime();
		if (goal_time - 0.001 < time && time < goal_time + 0.001){
		//printf("goal_time:%f \ttime: %f\n", goal_time, time);
		mRewardBuffer_ = r_strike;
		}
	}

	mRewardLog.Add(reward);mIRewardLog.Add(r_imitate);mSRewardLog.Add(r_strike);
}

void cSceneAug::GetRewardLog(cCircularBuffer<double> &rlog, cCircularBuffer<double> &rlog_imitate, cCircularBuffer<double> &rlog_strike) const
{
  rlog = mRewardLog; rlog_imitate = mIRewardLog; rlog_strike = mSRewardLog;
}

void cSceneAug::CheckGoalCleared()
{
	auto ctrl = dynamic_cast<cCtController*>(GetCharacter()->GetController().get());
	cSimGoal goal = ctrl->GetGoal(ctrl->GetTime());
	if(ctrl->GetGoalFlagTime() == 1){

	}else{
		auto kin_char = GetKinChar();
		double mTime = kin_char->GetTime();
		double GoalTime = goal.GetPos()[3];
		if(mTime > GoalTime)
		{
			///For visualizing when is GoalTime
			ctrl->SetGoalFlagTime(1);
		}
	}

	if(ctrl->GetGoalFlagClear() == 1){

	}else{
		int idx = ctrl->ChoosePertitionIdx();
		int ikjoint = mIKJoint[idx];
		tVector pos_ef = GetCharacter()->GetBodyPart(ikjoint)->GetPos();
		tVector pos_goal = goal.GetPos();
		const auto& kin_char = GetKinChar();
		pos_ef[3] = 0; pos_goal[3] = 0;
		if(sqrt((pos_ef - pos_goal).squaredNorm()) < goal.GetRadius()){
			ctrl->SetGoalFlagClear(1);
		}	
	}

}

void cSceneAug::UpdateKinChar(double timestep)
{
	const auto& kin_char = GetKinChar();
	double prev_phase = kin_char->GetPhase();
	kin_char->Update(timestep);
	double curr_phase = kin_char->GetPhase();

	if (curr_phase < prev_phase)
	{
		const auto& sim_char = GetCharacter();
		SyncKinCharNewCycle(*sim_char, *kin_char);
	}
}

void cSceneAug::ResetCharacters()
{
	int num_chars = GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		const auto& curr_char = GetCharacter(i);
		const auto& ctrl = dynamic_cast<cCtController*>(GetCharacter()->GetController().get());
		//ctrl->SampleGoalPosCuboid()
		if (mIKJoint.size() > 0){
			ctrl->SampleGoalPos();
			std::vector<cSimGoal> goal = ctrl->GetGoal();
			std::vector<tVector> goal_pos = {};
			for (int i = 0; i < goal.size(); i++){
				goal_pos.push_back(goal[i].GetPos());
			}
			BuildKinChar(goal_pos);
		}else if(mSample){
			BuildKinCharSample();
		}else{
			BuildKinChar();
		}
		
		ctrl->SetCyclePeriod(mKinChar->GetMotionDuration());
		ctrl->SetKinChar(mKinChar);
		curr_char->Reset();
	}

	ResetKinChar();
	if (EnableSyncChar())
	{
		SyncCharacters();
	}
}

void cSceneAug::ResetKinChar()
{
	double rand_time = 0;
	if(mMode==eModeTest){
		//rand_time = mRand.RandDouble(0.0, 0.3);
		rand_time = mRand.RandDouble(0.0, 0.1);
		//rand_time = 0;
		std::cout << "StartFromBeginning" << std::endl;
	}else if(mMode==eModeTrain){
		rand_time = CalcRandKinResetTime();
	}

	const cSimCharacter::tParams& char_params = mCharParams[0];
	const auto& kin_char = GetKinChar();

	kin_char->Reset();
	kin_char->SetOriginRot(tQuaternion::Identity());
	kin_char->SetOriginPos(char_params.mInitPos); // reset origin
	mInitTime = rand_time;
	kin_char->SetTime(rand_time);
	kin_char->Pose(rand_time);

	if (EnabledRandRotReset())
	{
		double rand_theta = mRand.RandDouble(-M_PI, M_PI);
		kin_char->RotateOrigin(cMathUtil::EulerToQuaternion(tVector(0, rand_theta, 0, 0)));
	}
}

void cSceneAug::SyncCharacters()
{
	const auto& kin_char = GetKinChar();
	const Eigen::VectorXd& pose = kin_char->GetPose();
	const Eigen::VectorXd& vel = kin_char->GetVel();
	
	const auto& sim_char = GetCharacter();
	sim_char->SetPose(pose);
	sim_char->SetVel(vel);

	const auto& ctrl = sim_char->GetController();
	auto ct_ctrl = dynamic_cast<cCtController*>(ctrl.get());
	if (ct_ctrl != nullptr)
	{
		double kin_time = GetKinTime();
		ct_ctrl->SetInitTime(kin_time);
	}
}

bool cSceneAug::EnableSyncChar() const
{
	const auto& kin_char = GetKinChar();
	return kin_char->HasMotion();
}

void cSceneAug::InitCharacterPosFixed(const std::shared_ptr<cSimCharacter>& out_char)
{
	// nothing to see here
}

void cSceneAug::InitJointWeights()
{
	CalcJointWeights(GetCharacter(), mJointWeights);
}

void cSceneAug::ResolveCharGroundIntersect()
{
	cRLSceneSimChar::ResolveCharGroundIntersect();

	if (EnableSyncChar())
	{
		SyncKinCharRoot();
	}
}

void cSceneAug::ResolveCharGroundIntersect(const std::shared_ptr<cSimCharacter>& out_char) const
{
	cRLSceneSimChar::ResolveCharGroundIntersect(out_char);
}

void cSceneAug::SyncKinCharRoot()
{
	const auto& sim_char = GetCharacter();
	tVector sim_root_pos = sim_char->GetRootPos();
	double sim_heading = sim_char->CalcHeading();

	const auto& kin_char = GetKinChar();
	double kin_heading = kin_char->CalcHeading();

	tQuaternion drot = tQuaternion::Identity();
	if (mSyncCharRootRot)
	{
		drot = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), sim_heading - kin_heading);
	}

	kin_char->RotateRoot(drot);
	kin_char->SetRootPos(sim_root_pos);
}

void cSceneAug::SyncKinCharNewCycle(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const
{
	if (mSyncCharRootRot)
	{
		double sim_heading = sim_char.CalcHeading();
		double kin_heading = out_kin_char.CalcHeading();
		tQuaternion drot = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), sim_heading - kin_heading);
		out_kin_char.RotateRoot(drot);
	}

	if (mSyncCharRootPos)
	{
		tVector sim_root_pos = sim_char.GetRootPos();
		tVector kin_root_pos = out_kin_char.GetRootPos();
		kin_root_pos[0] = sim_root_pos[0];
		kin_root_pos[2] = sim_root_pos[2];

		tVector origin = out_kin_char.GetOriginPos();
		double dh = kin_root_pos[1] - origin[1];
		double ground_h = mGround->SampleHeight(kin_root_pos);
		kin_root_pos[1] = ground_h + dh;

		out_kin_char.SetRootPos(kin_root_pos);
	}
}

double cSceneAug::GetKinTime() const
{
	const auto& kin_char = GetKinChar();
	return kin_char->GetTime();
}

bool cSceneAug::CheckKinNewCycle(double timestep) const
{
	bool new_cycle = false;
	const auto& kin_char = GetKinChar();
	if (kin_char->GetMotion().EnableLoop())
	{
		double cycle_dur = kin_char->GetMotionDuration();
		double time = GetKinTime();
		new_cycle = cMathUtil::CheckNextInterval(timestep, time, cycle_dur);
	}
	return new_cycle;
}


bool cSceneAug::HasFallen(const cSimCharacter& sim_char) const
{
	bool fallen = cRLSceneSimChar::HasFallen(sim_char);
	if (mEnableRootRotFail)
	{
		fallen |= CheckRootRotFail(sim_char);
	}

	return fallen;
}

bool cSceneAug::CheckRootRotFail(const cSimCharacter& sim_char) const
{
	const auto& kin_char = GetKinChar();
	bool fail = CheckRootRotFail(sim_char, *kin_char);
	return fail;
}

bool cSceneAug::CheckRootRotFail(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{
	const double threshold = 0.5 * M_PI;

	tQuaternion sim_rot = sim_char.GetRootRotation();
	tQuaternion kin_rot = kin_char.GetRootRotation();
	double rot_diff = cMathUtil::QuatDiffTheta(sim_rot, kin_rot);
	return rot_diff > threshold;
}

double cSceneAug::CalcRandKinResetTime()
{
	const auto& kin_char = GetKinChar();
	double dur = kin_char->GetMotionDuration();
	double rand_time = cMathUtil::RandDouble(0, dur);
	return rand_time;
}

cSimGoal cSceneAug::GetGoal()
{
	const auto& ctrl = dynamic_cast<cCtController*>(GetCharacter()->GetController().get());
	return ctrl->GetGoal(ctrl->GetTime());
}

double cSceneAug::GetHoldEndFrame() const
{
	if (mMode == eModeTrain){
		return mHoldEndFrame;
	} else if (mMode == eModeTest){
		return 0.0;
	}
}
