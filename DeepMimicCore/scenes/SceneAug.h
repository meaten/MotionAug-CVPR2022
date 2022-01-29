#pragma once

#include "scenes/RLSceneSimChar.h"
#include "anim/KinCharacter.h"

#include <Python.h>
#include <vector>

class cSceneAug : virtual public cRLSceneSimChar
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cSceneAug();
	virtual ~cSceneAug();

	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();
	virtual void InitPython();
	virtual void InitSampler();
	virtual void InitJava();

	virtual const std::shared_ptr<cKinCharacter>& GetKinChar() const;
	virtual void EnableRandRotReset(bool enable);
	virtual bool EnabledRandRotReset() const;

	virtual double CalcReward(int agent_id) const;
	virtual eTerminate CheckTerminate(int agent_id) const;

    virtual bool CheckRewardLog();
	virtual bool StartedNearBeginning();
	virtual Eigen::VectorXd GetSimAbsPose(const cSimCharacter& sim_char) const;
	virtual Eigen::VectorXd GetKinAbsPose(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const;
	virtual void UpdateLog();
	virtual cSimGoal GetGoal();
	virtual int PyInit();
	virtual void ResetPoseLog();
	virtual PyObject* Matrix2PyList(const Eigen::Matrix<double, 16, 60> mat) const;
	PyObject* std_vector_to_py_list(const std::vector<double>& v) const;
	virtual std::string GetName() const;

	void GetRewardLog(cCircularBuffer<double> &rlog, cCircularBuffer<double> &rlog_imitate, cCircularBuffer<double> &rlog_strike) const;

protected:

	std::string mMotionFile;
	std::string mMotionString;
	std::shared_ptr<cKinCharacter> mKinChar;

	Eigen::VectorXd mJointWeights;
	bool mEnableRandRotReset;
	bool mSyncCharRootPos;
	bool mSyncCharRootRot;
	bool mEnableRootRotFail;
	double mHoldEndFrame;
	bool mEarlyTermination=true;

	std::string mAugMode;

	// IK aug
	bool mEnableStrikeReward;
	std::string mGoalShape;
	
	std::vector<int> mIKJoint = {};
	std::vector<int> mKeyframe = {};
	std::vector<double> mPertitionTime = {};
	double mWindowTime = 0.4;

	// Goal sampling
    std::vector<tVector> mCenter = {};
    std::vector<double> mCenterT = {};

	std::vector<double> mSampleParamPlus = {0.0, 0.0, 0.0, 0.0};
	std::vector<double> mSampleParamMinus = {0.0, 0.0, 0.0, 0.0};

	double mInitTime;
  
	int mRewardType = 2;
	double mScaleGoalReward = 4.0;
	double mTimeFactorPow = 1.0;

	double mRewardBuffer;
    double mRewardBuffer_;

	bool mStartFromBeginning;

	PyObject *mSampler;
	bool mSample = false;
	std::string mClass;
	std::vector<std::string> mSubject;
	std::string mSamplerArgFile;
	
	cCircularBuffer<double> mRewardLog;
	cCircularBuffer<double> mIRewardLog;
	cCircularBuffer<double> mSRewardLog;
	
	Eigen::Matrix<double, 16, 60> mSimPoseLog;
	Eigen::Matrix<double, 16, 60> mKinPoseLog;

	virtual void Reset();

	virtual bool BuildCharacters();

	virtual void CalcJointWeights(const std::shared_ptr<cSimCharacter>& character, Eigen::VectorXd& out_weights) const;
	virtual bool BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCtController>& out_ctrl);
	virtual void BuildKinChar();
	virtual void BuildKinChar(std::vector<tVector> GoalPos);
	virtual void BuildKinCharSample();
	virtual bool BuildKinCharacter(int id, std::shared_ptr<cKinCharacter>& out_char) const;
	virtual bool BuildKinCharacter(int id, std::shared_ptr<cKinCharacter>& out_char, std::vector<tVector> GoalPos) const;
	virtual void UpdateCharacters(double timestep);
	virtual void UpdateKinChar(double timestep);

	virtual void ResetCharacters();
	virtual void ResetKinChar();
	virtual void SyncCharacters();
	virtual bool EnableSyncChar() const;
	virtual void InitCharacterPosFixed(const std::shared_ptr<cSimCharacter>& out_char);

	virtual double GetHoldEndFrame() const;
	virtual void CheckGoalCleared();

	virtual void InitJointWeights();
	virtual void ResolveCharGroundIntersect();
	virtual void ResolveCharGroundIntersect(const std::shared_ptr<cSimCharacter>& out_char) const;
	virtual void SyncKinCharRoot();
	virtual void SyncKinCharNewCycle(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const;

	virtual double GetKinTime() const;
	virtual bool CheckKinNewCycle(double timestep) const;
	virtual bool HasFallen(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const;
	
	virtual double CalcRandKinResetTime();
	virtual double CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& ref_char) const;
	virtual double CalcRewardGoal(const cSimCharacter& sim_char, const cKinCharacter& ref_char) const;
	virtual void CalcReward(const cSimCharacter& sim_char, const cKinCharacter& ref_char, double &r, double &r_i, double &r_s) const;
	
};
