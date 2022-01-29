#pragma once

#include "sim/CharController.h"
#include "util/CircularBuffer.h"
#include "sim/SimGoal.h"

class cDeepMimicCharController : public cCharController
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	virtual ~cDeepMimicCharController();

	virtual void Init(cSimCharacter* character, const std::string& param_file);
	virtual void Reset();
	virtual void Clear();

	virtual void Update(double timestep);
	virtual void PostUpdate(double timestep);
	virtual void UpdateCalcTau(double timestep, Eigen::VectorXd& out_tau);
	virtual void UpdateApplyTau(const Eigen::VectorXd& tau);

	virtual bool NeedNewAction() const;
	virtual void ApplyAction(const Eigen::VectorXd& action);
	virtual void RecordState(Eigen::VectorXd& out_state);
	virtual void RecordGoal(Eigen::VectorXd& out_goal) const;
	virtual void RecordAction(Eigen::VectorXd& out_action) const;

	virtual eActionSpace GetActionSpace() const;
	virtual int GetStateSize() const;
	virtual int GetGoalSize() const;
	virtual tVector GetGoalPos() const;
	virtual bool GetGoalFlagClear() const;
	virtual bool GetGoalFlagTime() const;

	virtual void SetGoalShape(std::string str);
	virtual void SetGoalPos(std::vector<tVector> goal_pos);
	virtual void SampleGoalPos();
    virtual void SampleGoalPosCuboid();
    virtual void SampleGoalPosEllipsoid();
	virtual void SampleGoalPosFanShape();
	virtual void SampleGoalPosFixed();
	virtual void SetGoalMin(tVector min);
	virtual void SetGoalMax(tVector max);
	virtual void SetGoalCenter(std::vector<tVector> center);
	virtual void SetPertitionTime(std::vector<double> pertition_time);
	virtual void SetGoalEllipsoidMinus( tVector minus );
	virtual void SetGoalEllipsoidPlus(  tVector plus  );
	virtual std::string GetGoalShape() const;
	virtual int GetGoalNum() const;
	virtual tVector GetGoalMin() const;
	virtual tVector GetGoalMax() const;
	virtual std::vector<tVector> GetGoalCenter() const;
	virtual tVector GetGoalCenter(double time) const;
	virtual tVector GetGoalEllipsoidMinus() const;
	virtual tVector GetGoalEllipsoidPlus() const;
	virtual void UpdateGoalEllipsoid();

	virtual int ChoosePertitionIdx() const;
	virtual int ChoosePertitionIdx(double time) const;

	virtual void SetRootVec(std::vector<tVector> root_vec_arr);
	virtual void SetGoalFanShapeHigh(tVector high);
	virtual void SetGoalFanShapeLow(tVector low);
	virtual std::vector<tVector> GetRootVec() const;
	virtual tVector GetGoalFanShapeHigh() const;
	virtual tVector GetGoalFanShapeLow() const;
	virtual tVector CalcGoalPosFanShape(double r, double h, double phi);
	virtual tVector CalcGoalPosFanShape(double r, double h, double phi, tVector center, tVector root_vec);

	virtual std::vector<cSimGoal> GetGoal() const;
	virtual cSimGoal GetGoal(double time) const;
	virtual void SetGoalFlagClear(int flag);
	virtual void SetGoalFlagTime(int flag);

	virtual double GetRewardMin() const;
	virtual double GetRewardMax() const;
	
	virtual void SetViewDistMin(double dist);
	virtual void SetViewDistMax(double dist);
	virtual double GetViewDistMin() const;
	virtual double GetViewDistMax() const;
	
	virtual void GetViewBound(tVector& out_min, tVector& out_max) const;

	virtual double GetPrevActionTime() const;
	virtual const tVector& GetPrevActionCOM() const;
	virtual double GetTime() const;

	virtual const Eigen::VectorXd& GetTau() const;
	virtual const cCircularBuffer<double>& GetValLog() const;
	virtual void LogVal(double val);

protected:
	double mTime;
	bool mNeedNewAction;
	Eigen::VectorXd mAction;
	Eigen::VectorXd mTau;

	double mViewDistMin;
	double mViewDistMax;
	int mPosDim;

	double mPrevActionTime;
	tVector mPrevActionCOM;

	bool mGoalPosInput;
	bool mPhasePosInput;
	std::string mGoalShape;
	std::vector<cSimGoal> mGoal;

	tVector mGoalMin = tVector(0.0, 0.0, 0.0, 0.0);
	tVector mGoalMax = tVector(0.0, 0.0, 0.0, 0.0);
	std::vector<tVector> mGoalCenter;
	std::vector<double> mPertitionTime;
	tVector mGoalEllipsoidMinus;
	tVector mGoalEllipsoidPlus;
	std::vector<tVector> mRootVec;
	tVector mGoalFanShapeHigh;
	tVector mGoalFanShapeLow;
	
	// for recording prediction from the value function, mainly for visualization
	cCircularBuffer<double> mValLog; 

	cDeepMimicCharController();

	virtual bool ParseParams(const Json::Value& json);

	virtual void ResetParams();
	virtual void InitResources();
	virtual void InitAction();
	virtual void InitTau();
	
	virtual int GetPosDim() const;

	virtual bool CheckNeedNewAction(double timestep) const;
	virtual void NewActionUpdate();
	virtual void HandleNewAction();
	virtual void PostProcessAction(Eigen::VectorXd& out_action) const;

	virtual void BuildStatePose(Eigen::VectorXd& out_pose) const;
	virtual void BuildStateVel(Eigen::VectorXd& out_vel) const;
	virtual int GetStatePoseOffset() const;
	virtual int GetStateVelOffset() const;
	virtual int GetStatePoseSize() const;
	virtual int GetStateVelSize() const;	
};
