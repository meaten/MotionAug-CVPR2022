#include "DeepMimicCharController.h"
#include "sim/SimCharacter.h"
#include <iostream>
#include <ctime>
#include <unistd.h>
#include <bits/stdc++.h>
#include "util/json/json.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"

const int gValLogSize = 1000;
const std::string gViewDistMinKey = "ViewDistMin";
const std::string gViewDistMaxKey = "ViewDistMax";
const std::string gGoalPosInput = "GoalPosInput";
///const std::string gPhasePosInput = "PhasePosInput";

cDeepMimicCharController::cDeepMimicCharController() : cCharController()
{
	mTime = 0;
	mPosDim = 0;
	SetViewDistMin(-0.5);
	SetViewDistMax(10);

	mPrevActionTime = mTime;
	mPrevActionCOM.setZero();

	mValLog.Reserve(gValLogSize);
}

cDeepMimicCharController::~cDeepMimicCharController()
{
}

void cDeepMimicCharController::Init(cSimCharacter* character, const std::string& param_file)
{
	cCharController::Init(character);
	
	LoadParams(param_file);

	ResetParams();

	mPosDim = GetPosDim();
	InitResources();
	
	mValid = true;
	mGoal = {};
	SampleGoalPosCuboid();
}

void cDeepMimicCharController::Reset()
{
	cCharController::Reset();
	ResetParams();
	NewActionUpdate();
}

void cDeepMimicCharController::Clear()
{
	cCharController::Clear();
	ResetParams();
}

void cDeepMimicCharController::Update(double time_step)
{
	cCharController::Update(time_step);
	UpdateCalcTau(time_step, mTau);
	UpdateApplyTau(mTau);
}

void cDeepMimicCharController::PostUpdate(double timestep)
{
	mNeedNewAction = CheckNeedNewAction(timestep);
	if (mNeedNewAction)
	{
		NewActionUpdate();
	}
}

void cDeepMimicCharController::UpdateCalcTau(double timestep, Eigen::VectorXd& out_tau)
{
	mTime += timestep;
	if (mNeedNewAction)
	{
		HandleNewAction();
	}
}

void cDeepMimicCharController::UpdateApplyTau(const Eigen::VectorXd& tau)
{
	mTau = tau;
	mChar->ApplyControlForces(tau);
}

void cDeepMimicCharController::SetViewDistMin(double dist)
{
	mViewDistMin = dist;
}

void cDeepMimicCharController::SetViewDistMax(double dist)
{
	mViewDistMax = dist;
}

double cDeepMimicCharController::GetViewDistMin() const
{
	return mViewDistMin;
}

double cDeepMimicCharController::GetViewDistMax() const
{
	return mViewDistMax;
}

void cDeepMimicCharController::GetViewBound(tVector& out_min, tVector& out_max) const
{
	tVector origin = mChar->GetRootPos();
	double max_len = mViewDistMax;
	out_min = origin - tVector(max_len, 0, max_len, 0);
	out_max = origin + tVector(max_len, 0, max_len, 0);
}

double cDeepMimicCharController::GetPrevActionTime() const
{
	return mPrevActionTime;
}

const tVector& cDeepMimicCharController::GetPrevActionCOM() const
{
	return mPrevActionCOM;
}

double cDeepMimicCharController::GetTime() const
{
	return mTime;
}

const Eigen::VectorXd& cDeepMimicCharController::GetTau() const
{
	return mTau;
}

const cCircularBuffer<double>& cDeepMimicCharController::GetValLog() const
{
	return mValLog;
}

void cDeepMimicCharController::LogVal(double val)
{
	mValLog.Add(val);
}

bool cDeepMimicCharController::NeedNewAction() const
{
	return mNeedNewAction;
}

void cDeepMimicCharController::ApplyAction(const Eigen::VectorXd& action)
{
	assert(action.size() == GetActionSize());
	mAction = action;
	PostProcessAction(mAction);
}

void cDeepMimicCharController::RecordState(Eigen::VectorXd& out_state)
{
	int state_size = GetStateSize();
	// fill with nans to make sure we don't forget to set anything
	out_state = std::numeric_limits<double>::quiet_NaN() * Eigen::VectorXd::Ones(state_size);

	Eigen::VectorXd ground;
	Eigen::VectorXd pose;
	Eigen::VectorXd vel;
	BuildStatePose(pose);
	BuildStateVel(vel);

	int pose_offset = GetStatePoseOffset();
	int pose_size = GetStatePoseSize();
	int vel_offset = GetStateVelOffset();
	int vel_size = GetStateVelSize();

	out_state.segment(pose_offset, pose_size) = pose;
	out_state.segment(vel_offset, vel_size) = vel;
}

void cDeepMimicCharController::RecordGoal(Eigen::VectorXd& out_goal) const
{
	int goal_size = GetGoalSize();
	out_goal = std::numeric_limits<double>::quiet_NaN() * Eigen::VectorXd::Ones(goal_size);

	if (goal_size != 0)
	{
		tVector goal = GetGoalPos(); //(x,y,z,t)
		goal[3] = mTime - goal[3];
		//printf("%lf\n%lf\n%lf\n%lf\n%d\n\n", goal[0], goal[1], goal[2], goal[3], GetGoalFlag());sleep(0.1);
		out_goal.segment(0, 4) = goal;
		///now we don't consider flag
		//out_goal.segment(4, 1) = GetGoalFlag() * Eigen::VectorXd::Ones(1);
		out_goal.segment(4, 1) = 0 * Eigen::VectorXd::Ones(1);
	}
}

eActionSpace cDeepMimicCharController::GetActionSpace() const
{
	return eActionSpaceContinuous;
}

void cDeepMimicCharController::RecordAction(Eigen::VectorXd& out_action) const
{
	out_action = mAction;
}

int cDeepMimicCharController::GetStateSize() const
{
	int state_size = 0;
	state_size += GetStatePoseSize();
	state_size += GetStateVelSize();
	return state_size;
}


int cDeepMimicCharController::GetGoalSize() const
{
	if(mGoal.size() > 0 && mGoalPosInput)
	{
		return mGoal[0].GetSize();
	}
	else
	{
		return 0;	
	}
	
}

tVector cDeepMimicCharController::GetGoalPos() const
{
	int idx = ChoosePertitionIdx();
	return mGoal[idx].GetPos();
}

bool cDeepMimicCharController::GetGoalFlagClear() const
{
	int idx = ChoosePertitionIdx();
	return mGoal[idx].GetFlagClear();
}
bool cDeepMimicCharController::GetGoalFlagTime() const
{
	int idx = ChoosePertitionIdx();
	return mGoal[idx].GetFlagTime();
}

double cDeepMimicCharController::GetRewardMin() const
{
	return 0;
}

double cDeepMimicCharController::GetRewardMax() const
{
	return 1;
}

bool cDeepMimicCharController::ParseParams(const Json::Value& json)
{
	bool succ = cCharController::ParseParams(json);

	mViewDistMin = json.get(gViewDistMinKey, mViewDistMin).asDouble();
	mViewDistMax = json.get(gViewDistMaxKey, mViewDistMax).asDouble();
	mGoalPosInput = json.get(gGoalPosInput, mGoalPosInput).asBool();
	///mPhasePosInput = json.get(gPhasePosInput, mPhasePosInput).asBool();
	

	return succ;
}

void cDeepMimicCharController::ResetParams()
{
	mTime = 0;
	mNeedNewAction = true;
	mTau.setZero();
	mValLog.Clear();

	mPrevActionTime = mTime;
	mPrevActionCOM.setZero();
}

void cDeepMimicCharController::InitResources()
{
	InitAction();
	InitTau();
}

void cDeepMimicCharController::InitAction()
{
	mAction = Eigen::VectorXd::Zero(GetActionSize());
}

void cDeepMimicCharController::InitTau()
{
	mTau = Eigen::VectorXd::Zero(mChar->GetNumDof());
}

int cDeepMimicCharController::GetPosDim() const
{
	int dim = 3;
	return dim;
}

bool cDeepMimicCharController::CheckNeedNewAction(double timestep) const
{
	return false;
}

void cDeepMimicCharController::NewActionUpdate()
{
}

void cDeepMimicCharController::HandleNewAction()
{
	mPrevActionTime = mTime;
	mPrevActionCOM = mChar->CalcCOM();
	mNeedNewAction = false;
}

void cDeepMimicCharController::PostProcessAction(Eigen::VectorXd& out_action) const
{
}

void cDeepMimicCharController::BuildStatePose(Eigen::VectorXd& out_pose) const
{
	tMatrix origin_trans = mChar->BuildOriginTrans();

	tVector root_pos = mChar->GetRootPos();
	tVector root_pos_rel = root_pos;

	root_pos_rel[3] = 1;
	root_pos_rel = origin_trans * root_pos_rel;
	root_pos_rel[3] = 0;

	out_pose = Eigen::VectorXd::Zero(GetStatePoseSize());
	out_pose[0] = root_pos_rel[1];
	
	int idx = 1;
	int num_parts = mChar->GetNumBodyParts();
	for (int i = 1; i < num_parts; ++i)
	{
		if (mChar->IsValidBodyPart(i))
		{
			const auto& curr_part = mChar->GetBodyPart(i);
			tVector curr_pos = curr_part->GetPos();

			curr_pos[3] = 1;
			curr_pos = origin_trans * curr_pos;
			curr_pos[3] = 0;
			curr_pos -= root_pos_rel;

			out_pose.segment(idx, mPosDim) = curr_pos.segment(0, mPosDim);
			idx += mPosDim;
		}
	}
}

void cDeepMimicCharController::BuildStateVel(Eigen::VectorXd& out_vel) const
{
	out_vel.resize(GetStateVelSize());
	tMatrix origin_trans = mChar->BuildOriginTrans();

	tVector root_pos = mChar->GetRootPos();
	
	int idx = 0;
	int num_parts = mChar->GetNumBodyParts();
	for (int i = 0; i < num_parts; ++i)
	{
		tVector curr_vel = mChar->GetBodyPartVel(i);
		curr_vel = origin_trans * curr_vel;
		out_vel.segment(idx, mPosDim) = curr_vel.segment(0, mPosDim);
		idx += mPosDim;
	}
}

int cDeepMimicCharController::GetStatePoseOffset() const
{
	return 0;
}

int cDeepMimicCharController::GetStateVelOffset() const
{
	return GetStatePoseOffset() + GetStatePoseSize();
}

int cDeepMimicCharController::GetStatePoseSize() const
{
	return mChar->GetNumBodyParts() * mPosDim - 1; // -1 for root x
}

int cDeepMimicCharController::GetStateVelSize() const
{
	return mChar->GetNumBodyParts() * mPosDim;
}

void cDeepMimicCharController::SampleGoalPos()
{
	if (mGoalShape=="Ellipsoid")
	{
		SampleGoalPosEllipsoid();
	}else if (mGoalShape=="FanShape")
	{
		SampleGoalPosFanShape();
	}else if (mGoalShape=="Fixed")
	{
		SampleGoalPosFixed();
	}else{
		
	}
}

void cDeepMimicCharController::SampleGoalPosCuboid()
{
	std::vector<tVector> center = GetGoalCenter();

	std::vector<tVector> goal_pos_arr = {};
	for (int i = 0; i < GetGoalNum(); i++ ){
		tVector rel_pos = tVector(cMathUtil::RandDouble(mGoalMin[0], mGoalMax[0]), 
								cMathUtil::RandDouble(mGoalMin[1], mGoalMax[1]), 
								cMathUtil::RandDouble(mGoalMin[2], mGoalMax[2]), 0.0);
		double time = cMathUtil::RandDouble(mGoalMin[3], mGoalMax[3]);
		tVector goal_pos = center[i] + tVector(rel_pos[0], rel_pos[1] , rel_pos[2], time); //xzy
		goal_pos_arr.push_back(goal_pos);
	}
	SetGoalPos(goal_pos_arr);
}

void cDeepMimicCharController::SampleGoalPosEllipsoid()
{
	std::vector<tVector> center = GetGoalCenter();

	std::vector<tVector> goal_pos_arr = {};
	for (int i = 0; i < GetGoalNum(); i++ ){
		tVector randvec = tVector(cMathUtil::RandDouble(0, 1),
		cMathUtil::RandDouble(-1, 1),
		cMathUtil::RandDouble(0, 2*M_PI),
		cMathUtil::RandDouble(-1, 1)
		);

		double r = std::cbrt(randvec[0]);
		double d = r * std::sqrt(1-std::pow(randvec[1], 2)) * std::cos(randvec[2]);
		double h = r * std::sqrt(1-std::pow(randvec[1], 2)) * std::sin(randvec[2]);
		double y = r * randvec[1];
		double t = randvec[3];

		randvec = tVector(d, h, y, t);

		for (int i = 0; i < 4; ++i)
		{
			if (std::signbit(randvec[i]))
			{
				randvec[i] *= mGoalEllipsoidMinus[i];
			}else{
				randvec[i] *= mGoalEllipsoidPlus[i];
			}
		}

		goal_pos_arr.push_back(randvec + center[i]);
	}
	SetGoalPos(goal_pos_arr);
}

void cDeepMimicCharController::SampleGoalPosFanShape()
{
	std::vector<tVector> center = GetGoalCenter();
	std::vector<tVector> root_vec = GetRootVec();

	std::vector<tVector> goal_pos_arr = {};
	for (int i = 0; i < GetGoalNum(); i++ ){
		double r = cMathUtil::RandDouble(mGoalFanShapeLow[0], mGoalFanShapeHigh[0]);
		double h = cMathUtil::RandDouble(mGoalFanShapeLow[1], mGoalFanShapeHigh[1]);
		double phi = cMathUtil::RandDouble(mGoalFanShapeLow[2], mGoalFanShapeHigh[2]);
		double delta_t = cMathUtil::RandDouble(mGoalFanShapeLow[3], mGoalFanShapeHigh[3]);

		double t = center[i][3] + delta_t;

		tVector goal = CalcGoalPosFanShape(r, h, phi, center[i], root_vec[i]);
		goal[3]	= t;

		goal_pos_arr.push_back(goal);
	}

	SetGoalPos(goal_pos_arr);
}

tVector cDeepMimicCharController::CalcGoalPosFanShape(double r, double h, double phi)
{
	int idx = ChoosePertitionIdx(mTime);
	return CalcGoalPosFanShape(r, h, phi, GetGoalCenter()[idx], GetRootVec()[idx]);
}

tVector cDeepMimicCharController::CalcGoalPosFanShape(double r, double h, double phi, tVector center, tVector root_vec)
{
	if (mGoalShape!="FanShape"){
		throw std::invalid_argument("invalid goal shape");
	}

	tVector r_center = tVector(center[0] - root_vec[0], 0.0, center[2] - root_vec[2], 0.0);
	double h_center = center[1];

	r *= r_center.norm();
	h *= h_center;
	
	//calc initial orientation along z axis
	tVector root_to_end = center - root_vec;
	double offset_phi = std::atan2(root_to_end[0], root_to_end[2]);
	//std::cerr << << std::endl;
	//std::cerr << offset_phi / 3.14 * 180.0 << std::endl;
	phi += offset_phi;

	double d = root_vec[0] + r * std::sin(phi);
	double y = root_vec[2] + r * std::cos(phi);

	return tVector(d, h, y, 0.0);
}

void cDeepMimicCharController::SampleGoalPosFixed()
{
	/*
	std::vector<double> data{
		-0.55030456,  0.17344523,  0.33853256, -0.45143815,  0.1344888 ,
        0.33196407,  0.39420045,  0.21175151,  0.67615969, -0.37132617,
        0.18777793,  0.52778275,  0.1100057 ,  0.2490166 ,  0.90678776,
       -0.65016306,  0.26301563,  0.48416693, -0.4372614 ,  0.1047825 ,
        0.4448875 , -0.58759834,  0.16028056,  0.5045032 ,  0.02320352,
       -0.16039327,  0.5216473 , -0.79096532, -0.01562117,  0.47136643,
        0.1812988 ,  0.22234013,  0.94534417, -0.44549859, -0.24202229,
        0.51314074, -0.94785809,  0.07800099,  0.31376388, -0.11969645,
        0.32789476,  0.85155034, -0.25269034,  0.19488823,  0.6024223 ,
       -0.62195563,  0.16136151,  0.46878293, -0.56417004,  0.18271547,
        0.4437708 ,  0.46939704,  0.0924977 ,  0.62259477, -0.4614151 ,
        0.07481184,  0.59777687, -0.07443924,  0.11528997,  0.65955701,
       -0.73615048,  0.11246111,  0.2594131 , -0.34477561,  0.10096736,
        0.37213148, -0.10559358,  0.01051216,  0.8691983 , -0.61170938,
        0.10772399,  0.67847731, -0.38517407,  0.07488243,  0.73402381,
        0.12418261,  0.03366975,  0.80250474,  0.32648072,  0.05739898,
        0.92445051, -0.02746969,  0.2313186 ,  0.53401196, -0.37381252,
        0.15751839,  0.68027722, -0.37211102, -0.72066181,  0.29625027,
       -0.69390406, -0.15952738,  0.49118527, -0.37262031,  0.05093357,
        0.62567488, -0.303164  ,  0.16623057,  0.34543883, -0.51562179,
        0.15447679,  0.62958435, -0.10529999,  0.12585499,  0.78522749,
        0.16724415,  0.22850502,  0.90339776,  0.1662715 ,  0.23543849,
        0.87151615, -0.63809297, -0.16912666,  0.51831714, -0.17689036,
        0.05185577,  0.22585132, -0.26943544,  0.01237446,  0.71008969,
       -0.55242149,  0.14575954,  0.0904428 ,  0.12177659,  0.12411295,
        0.79194863, -0.25653954,  0.04392869,  0.73298314, -0.84685588,
        0.05237586,  0.17618349, -0.68370475,  0.13782303,  0.22398373,
       -0.24560159,  0.13879344,  0.58921823, -0.07470679,  0.04839818,
        0.90446173, -0.28739394,  0.29015718,  0.82306786, -0.66190752,
        0.25698423,  0.59163189, -0.50094441,  0.02864806,  0.41395012,
       -0.66702094,  0.04516594,  0.12088108, -0.12529957,  0.0071082 ,
        0.63964069, -0.70011652,  0.02083142,  0.22254944,  0.00440999,
        0.04927434,  0.649926  , -0.31032199,  0.14492603,  0.61681888,
       -0.00801577,  0.06837391,  0.83548035, -0.09954765,  0.00395134,
        0.53560206, -0.09103094,  0.2658477 ,  0.745826  ,  0.09029046,
        0.11913215,  0.65725149, -0.29352172, -0.10767512,  0.52083493,
       -0.39494109,  0.00891359,  0.56557831,  0.17406788,  0.1220957 ,
        0.79338188, -0.43428819,  0.14987576,  0.49767126, -0.71476197,
        0.04540413,  0.61228037, -0.18425629,  0.1904873 ,  0.64125144,
       -0.59476131,  0.06417093, -0.02088459, -0.12870927,  0.11986671,
        0.85791497, -0.93242913,  0.14470337,  0.26338151, -0.30905911,
        0.07957557,  0.73260905, -0.14455376,  0.03684737,  0.69368958,
        0.52501307,  0.23097202,  0.59617453, -0.39109607,  0.16148749,
        0.72371734, -0.22496951,  0.13987619,  0.7378153 , -0.5849786 ,
        0.10278384,  0.62655248, -0.07375916,  0.19252109,  0.80308789,
        0.0313191 ,  0.13390264,  0.69891526,  0.26769122, -0.04987061,
        0.91923235, -0.69716657,  0.15179755,  0.14109545, -0.52931459,
        0.21367532,  0.35344224, -0.35726686,  0.01410543,  0.40489286,
        0.1988275 ,  0.0793077 ,  0.94917218,  0.15907669,  0.13440154,
        0.92213437,  0.09023314,  0.23866226,  0.67194879,  0.22520109,
        0.28095738,  0.68328229,  0.30549637,  0.19870722,  0.87975893,
        0.24273468, -0.08609089,  0.7470963 , -0.36518243, -0.00915629,
        0.63204979, -0.10852264,  0.13701241,  0.64099326, -0.07943753,
        0.19502203,  0.89793209,  0.09428831,  0.10150008,  0.90424959,
        0.052308  ,  0.09272412,  0.6390212 , -0.21058673,  0.07611456,
        0.78580265, -0.36246645, -0.09537415,  0.73488605, -0.65231491,
        0.20278102,  0.54955637, -0.13273606,  0.12281882,  0.81779777,
       -0.55513653,  0.04615852,  0.53032442, -0.6549183 , -0.01327782,
        0.47885679, -0.66736046,  0.02660939,  0.35151766, -0.74389781,
       -0.10716423,  0.36462264, -0.13963395,  0.07153922,  0.88194615,
       -0.89811116, -0.01318206,  0.01259567, -0.99244994,  0.19899163,
        0.0510629 , -0.7039131 ,  0.13163591,  0.42598626, -0.14358658,
        0.06346911,  0.54864282, -0.3574566 , -0.09406448,  0.60855466,
       -0.63502138, -0.15459214,  0.49894572, -0.7634931 ,  0.17511471,
        0.44954859, -0.55966409,  0.21799046,  0.32789087, -0.80490804,
        0.09504201,  0.24582148, -0.68317549,  0.08928825,  0.32203867,
       -0.83504614, -0.3834793 ,  0.05962773, -0.15183755,  0.09241624,
        0.51007744,  0.35513637,  0.11335064,  0.80781809,  0.0760715 ,
        0.1463513 ,  0.683487  , -0.54964666,  0.22700435,  0.49810197,
       -0.30538033,  0.17174808,  0.49235387, -0.77026702,  0.09129397,
        0.31909395, -0.67514274,  0.18794361,  0.40886154, -0.21019801,
        0.14310835,  0.7225987 , -0.39729039,  0.10107499,  0.63482419,
       -0.30512239,  0.1105073 ,  0.41049413,  0.18266345,  0.22534302,
        0.77737834, -0.68139174,  0.02281196,  0.29189411, -0.57533367,
       -0.04642999,  0.51429203, -0.231863  , -0.09723617,  0.60293179,
        0.06723648,  0.22070726,  0.56059495,  0.09298445, -0.2100961 ,
        0.87082412, -0.62511125,  0.26916661,  0.39638667, -0.20267067,
       -0.1171804 ,  0.69294011, -0.48182877,  0.10771831,  0.55815709,
       -0.50806268,  0.22604383,  0.42682381, -0.62230878,  0.20737499,
        0.21742616, -0.6499018 , -0.02392583,  0.31172013,  0.48148553,
        0.14106873,  0.78634788,  0.09917079,  0.17756499,  0.85599575,
       -0.24306027,  0.20905344,  0.64098506, -0.38338956,  0.1958946 ,
        0.57626416, -0.81141389,  0.03674275,  0.51978169,  0.81856301,
        0.15654056,  0.63402987,  0.08602489,  0.08080896,  0.8518111 ,
        0.2681075 ,  0.20620986,  0.77228658, -0.42880263,  0.1547928 ,
        0.73116004, -0.58470267,  0.1618839 ,  0.46641405, -0.07815801,
        0.18223523,  0.84298833, -0.27702232,  0.08311721,  0.63166607,
        0.2394036 ,  0.17081884,  0.83667564, -0.18201669,  0.07775184,
        0.80159331, -0.14166385,  0.24340904,  0.79036502,  0.76208945,
        0.28991138,  0.49649471, -0.43304138,  0.11482845,  0.43935098,
       -0.68644923,  0.20880749,  0.41528361, -0.59531695,  0.17919252,
        0.26603546, -0.70777603,  0.20236348,  0.53149855, -0.56986713,
        0.17053524,  0.58428513, -0.45415878,  0.15712151,  0.58436445,
       -0.05682733,  0.19407365,  0.66782276, -0.45096595,  0.21158886,
        0.48702994, -0.99439681, -0.26726131,  0.18983703,  0.08452483,
        0.16095019,  0.57680333, -0.21837345,  0.15746488,  0.49736639,
       -0.77661485,  0.20209652,  0.08819516, -0.86567878, -0.0541436 ,
       -0.01463012, -0.19890006,  0.05906639,  0.77883888, -0.14791055,
        0.21362718,  0.8014154 , -0.18986241,  0.17685051,  0.66434735,
       -0.05578792,  0.24714933,  0.57549833, -0.38689892,  0.09164143,
        0.76356509, -0.37278149,  0.279097  ,  0.60520235, -0.76212417,
        0.239433  ,  0.46866304, -0.1797832 ,  0.18804746,  0.55134005,
       -0.2894161 , -0.12949901,  0.57565926, -0.06990771,  0.06418605,
        0.76800968};
	*/
	std::vector<tVector> center = GetGoalCenter();
	std::vector<tVector> root_vec = GetRootVec();

	std::vector<tVector> goal_pos_arr = {};
	for (int i = 0; i < GetGoalNum(); i++ ){

		//int idx = std::rand() % 172;
	
		//double delta_t = cMathUtil::RandDouble(mGoalFanShapeLow[3], mGoalFanShapeHigh[3]);
		//double t = center[i][3] + delta_t;

		//tVector goal = tVector(data[idx * 3], data[idx * 3 + 1], data[idx * 3 + 2], 0.0) + root_vec[i];
		//goal[3] = t;
		tVector goal = center[i];
		goal[3] = center[i][3];
		goal_pos_arr.push_back(goal);
	}
	SetGoalPos(goal_pos_arr);
}

void cDeepMimicCharController::SetGoalPos(std::vector<tVector> goal_pos)
{
	assert(goal_pos.size() == mRootVec.size());
	mGoal = {};
	for (int i = 0; i < goal_pos.size(); i++){
		cSimGoal goal;
		goal.SetPos(goal_pos[i]);
		goal.SetFlagClear(0);
		goal.SetFlagTime(0);
		mGoal.push_back(goal);
	}
}


void cDeepMimicCharController::UpdateGoalEllipsoid()
{
	/*
	tVector goal_pos = GetGoalPos();
	
	tVector delta = goal_pos - mGoalCenter;
	double onepercentSphereVol = 1 - 0.117806;
	double stepsize = 0.01;
	
	for (int i = 0; i < 3; ++i)
	{
		if (std::signbit(delta[i]))
		{
			delta[i] /= mGoalEllipsoidMinus[i];
			if (std::abs(delta[i]) > onepercentSphereVol && mGoalEllipsoidMinus[i] < mGoalMin[i]){mGoalEllipsoidMinus[i] += stepsize;}
		}else{
			delta[i] /= mGoalEllipsoidPlus[i];
			if (delta[i]           > onepercentSphereVol && mGoalEllipsoidPlus[i]  < mGoalMax[i]){mGoalEllipsoidPlus[i] += stepsize;}
		}
	}

	if (std::signbit(delta[3]))
	{
		delta[3] /= mGoalEllipsoidMinus[3];
		if (std::abs(delta[3]) > 0.99 && std::abs(delta[3]) < mGoalCenter[3] / 2 && mGoalEllipsoidMinus[3] < mGoalMin[3]){mGoalEllipsoidMinus[3] += stepsize;}
	}else{
		delta[3] /= mGoalEllipsoidPlus[3];
		if (delta[3]           > 0.99 && mGoalEllipsoidPlus[3] < mGoalMax[3]){mGoalEllipsoidPlus[3] += stepsize;}
	}

	//printf("%f %f %f %f\n", t, mDeltaTminus, goal_pos[3], mCenterT);

	//std::cerr << mGoalMin << mGoalMax <<  std::endl;

	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
	mGoalEllipsoidMinus[0], mGoalEllipsoidMinus[1], mGoalEllipsoidMinus[2], mGoalEllipsoidMinus[3],
	mGoalEllipsoidPlus[0], mGoalEllipsoidPlus[1], mGoalEllipsoidPlus[2], mGoalEllipsoidPlus[3]);
	*/
}

void cDeepMimicCharController::SetGoalShape(std::string str)
{
	mGoalShape=str;
}

void cDeepMimicCharController::SetGoalMin(tVector min)
{
	mGoalMin = min;
}
void cDeepMimicCharController::SetGoalMax(tVector max)
{
	mGoalMax = max;
}
void cDeepMimicCharController::SetGoalCenter(std::vector<tVector> center)
{
	mGoalCenter = center;
}
void cDeepMimicCharController::SetPertitionTime(std::vector<double> pertition_time)
{
	mPertitionTime = pertition_time;
}
void cDeepMimicCharController::SetGoalEllipsoidMinus( tVector minus )
{
	mGoalEllipsoidMinus = minus;
}
void cDeepMimicCharController::SetGoalEllipsoidPlus(  tVector plus  )
{
	mGoalEllipsoidPlus = plus;
}
std::string cDeepMimicCharController::GetGoalShape() const
{
	return mGoalShape;
}
int cDeepMimicCharController::GetGoalNum() const
{
	return mRootVec.size();
}
tVector cDeepMimicCharController::GetGoalMin() const
{
	return mGoalMin;
}
tVector cDeepMimicCharController::GetGoalMax() const
{
	return mGoalMax;
}
std::vector<tVector> cDeepMimicCharController::GetGoalCenter() const
{
	return mGoalCenter;
}
tVector cDeepMimicCharController::GetGoalCenter(double time) const
{
	int idx = ChoosePertitionIdx(time);
	return mGoalCenter[idx];
}
tVector cDeepMimicCharController::GetGoalEllipsoidMinus() const
{
	return mGoalEllipsoidMinus;
}
tVector cDeepMimicCharController::GetGoalEllipsoidPlus() const
{
	return mGoalEllipsoidPlus;
}

void cDeepMimicCharController::SetRootVec(std::vector<tVector> root_vec_arr)
{
	mRootVec = root_vec_arr;
}

void cDeepMimicCharController::SetGoalFanShapeHigh(tVector high)
{
	mGoalFanShapeHigh = high;
}
void cDeepMimicCharController::SetGoalFanShapeLow(tVector low)
{
	mGoalFanShapeLow = low;
}

std::vector<tVector> cDeepMimicCharController::GetRootVec() const
{
	return mRootVec;
}
tVector cDeepMimicCharController::GetGoalFanShapeHigh() const
{
	return mGoalFanShapeHigh;
}
tVector cDeepMimicCharController::GetGoalFanShapeLow() const
{
	return mGoalFanShapeLow;
}

std::vector<cSimGoal> cDeepMimicCharController::GetGoal() const
{
	return mGoal;
}
								 
cSimGoal cDeepMimicCharController::GetGoal(double time) const
{
	int idx = ChoosePertitionIdx(time);
	return mGoal[idx];
}

void cDeepMimicCharController::SetGoalFlagClear(int flag)
{
	int idx = ChoosePertitionIdx();
	mGoal[idx].SetFlagClear(flag);
}
void cDeepMimicCharController::SetGoalFlagTime(int flag)
{
	int idx = ChoosePertitionIdx();
	mGoal[idx].SetFlagTime(flag);
}

int cDeepMimicCharController::ChoosePertitionIdx() const
{
	return ChoosePertitionIdx(mTime);
}

int cDeepMimicCharController::ChoosePertitionIdx(double time) const
{
	for (int i; i < mPertitionTime.size(); i++)
	{
		if (mTime > mPertitionTime[i] && i < mPertitionTime.size() - 1){
			continue;
		} else {
			return i;
		}
	}
}
