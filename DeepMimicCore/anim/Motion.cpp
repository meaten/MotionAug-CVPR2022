#include "Motion.h"
#include <assert.h>
#include <iostream>
#include <unistd.h>

#include "util/FileUtil.h"
#include "util/JsonUtil.h"

#include "Eigen/Dense"
#include "Eigen/Core"

#include "jni.h"

const double gMinTime = 0;

// Json keys
const std::string cMotion::gFrameKey = "Frames";
const std::string cMotion::gLoopKey = "Loop";
const std::string cMotion::gVelFilterCutoffKey = "VelFilterCutoff";
const std::string gRightJointsKey = "RightJoints";
const std::string gLeftJointsKey = "LeftJoints";

const std::string gLoopStr[cMotion::eLoopMax] =
{
	"none",
	"wrap",
	"mirror"
};

std::string cMotion::BuildFrameJson(const Eigen::VectorXd& frame)
{
	std::string json = cJsonUtil::BuildVectorJson(frame);
	return json;
}

cMotion::tParams::tParams()
{
	mMotionFile = "";
	mMotionString = "";
	mBlendFunc = nullptr;
	mMirrorFunc = nullptr;
	mVelFunc = nullptr;
	mPostProcessFunc = nullptr;

	mRightJoints.clear();
	mLeftJoints.clear();

	mCenter = {};
	mGoalPos = {};
	mIKJoint = {};
	mPertitionTime = {};
};

cMotion::cMotion()
{
	Clear();
	mLoop = eLoopNone;
	mVelFilterCutoff = 1; // cutoff 6 Hz
}

cMotion::~cMotion()
{

}

void cMotion::Clear()
{
	mFrames.resize(0, 0);
	mFrameVel.resize(0, 0);
	mFrameVelMirror.resize(0, 0);
	mParams = tParams();
	
}

bool cMotion::Load(const tParams& params)
{
	Clear();
	
	mParams = params;
	Json::Value root;
	Json::Reader reader;
	bool succ;
	if (params.mMotionString != ""){
		succ = reader.parse(params.mMotionString, root);
	}else{
		std::ifstream f_stream(mParams.mMotionFile);
		succ = reader.parse(f_stream, root);
		f_stream.close();
	}

	if (succ)
	{
		succ = LoadJson(root);
		if (succ)
		{
			PostProcessFrames(mFrames);
			if (params.mGoalPos.size() > 0){
				FabrikIK(mFrames, params.mGoalPos, params.mCenter, params.mIKJoint, params.mPertitionTime, params.mWindowTime);
				Warp(mFrames, params.mGoalPos, params.mCenter, params.mPertitionTime);
			}
			UpdateVel();
		}
		else
		{
			printf("Failed to load motion from file %s\n", mParams.mMotionFile.c_str());
			assert(false);
		}
	}
	else
	{
		printf("Failed to parse Json from %s\n", mParams.mMotionFile.c_str());
		assert(false);
	}
	return succ;
}

void cMotion::Init(int num_frames, int num_dofs)
{
	Clear();
	mFrames = Eigen::MatrixXd::Zero(num_frames, num_dofs + eFrameMax);
}

bool cMotion::IsValid() const
{
	return GetNumFrames() > 0;
}

int cMotion::GetNumDof() const
{
	return GetFrameSize() - eFrameMax;
}

int cMotion::GetNumFrames() const
{
	return static_cast<int>(mFrames.rows());
}

int cMotion::GetFrameSize() const
{
	return static_cast<int>(mFrames.cols());
}

void cMotion::BuildFrameVel(Eigen::MatrixXd& out_frame_vel, bool mirror /*= false*/) const
{
	int num_frames = GetNumFrames();
	int dof = GetNumDof();
	out_frame_vel = Eigen::MatrixXd::Zero(num_frames, dof);

	Eigen::VectorXd vel;
	for (int f = 0; f < num_frames - 1; ++f)
	{
		double dt = GetFrameDuration(f);
		Eigen::VectorXd frame0 = GetFrame(f);
		Eigen::VectorXd frame1 = GetFrame(f + 1);
		if (mirror)
		{
			MirrorFrame(frame0);
			MirrorFrame(frame1);
		}

		CalcFrameVel(frame0, frame1, dt, vel);
		out_frame_vel.row(f) = vel;

		//if(vel[1] > 100)
		//{
		//	printf("%lf\n%d\n\n\n", vel[1], f);sleep(0.1);
		//}
	}

	if (num_frames > 1)
	{
		out_frame_vel.row(num_frames - 1) = out_frame_vel.row(num_frames - 2);
	}

	if (mVelFilterCutoff > 0)
	{
		FilterFrameVel(out_frame_vel);
	}

	//velocity diverge to infinity after filtering on very long frame
	/*
	for (int f = 0; f < num_frames - 1; ++f)
	{
		vel = out_frame_vel.row(f);
		if(vel[1] > 100)
		{
			printf("%lf\n%d\n\n\n", vel[1], f);sleep(0.1);
		}

	}
	*/
}

void cMotion::FilterFrameVel(Eigen::MatrixXd& out_frame_vel) const
{
	double dt = GetFrameDuration(0);
	int num_dof = static_cast<int>(out_frame_vel.cols());

	for (int i = 0; i < num_dof; ++i)
	{
		Eigen::VectorXd x = out_frame_vel.col(i);
		cMathUtil::ButterworthFilter(dt, mVelFilterCutoff, x);
		out_frame_vel.col(i) = x;
	}
}

cMotion::tFrame cMotion::GetFrame(int i) const
{
	int frame_size = GetFrameSize();
	return mFrames.row(i).segment(eFrameMax, frame_size - eFrameMax);
}

void cMotion::SetFrame(int i, const tFrame& frame)
{
	int frame_size = GetFrameSize();
	assert(frame.size() == frame_size - eFrameMax);
	mFrames.row(i).segment(eFrameMax, frame_size - eFrameMax) = frame;
}

void cMotion::SetFrameTime(int i, double time)
{
	mFrames(i, 0) = time;
}

void cMotion::BlendFrames(int a, int b, double lerp, tFrame& out_frame) const
{
	lerp = cMathUtil::Saturate(lerp);

	// remove time params
	tFrame frame0 = GetFrame(a);
	tFrame frame1 = GetFrame(b);

	if (HasBlendFunc())
	{
		mParams.mBlendFunc(&frame0, &frame1, lerp, &out_frame);
	}
	else
	{
		BlendFramesIntern(&frame0, &frame1, lerp, &out_frame);
	}
}

void cMotion::CalcFrame(double time, tFrame& out_frame, bool force_mirror /*=false*/) const
{
	int idx;
	double phase;
	CalcIndexPhase(time, idx, phase);

	BlendFrames(idx, idx + 1, phase, out_frame);
	if (NeedMirrorFrame(time) || force_mirror)
	{
		MirrorFrame(out_frame);
	}
}

void cMotion::CalcFrameVel(double time, cMotion::tFrame& out_vel, bool force_mirror /*=false*/) const
{
	if (!EnableLoop() && time >= GetDuration())
	{
		out_vel = Eigen::VectorXd::Zero(GetNumDof());
	}
	else
	{
		const Eigen::MatrixXd* vel_frame = nullptr;
		if (NeedMirrorFrame(time) || force_mirror)
		{
			vel_frame = &mFrameVelMirror;
		}
		else
		{
			vel_frame = &mFrameVel;
		}

		int idx;
		double phase;
		CalcIndexPhase(time, idx, phase);
		auto vel0 = vel_frame->row(idx);
		auto vel1 = vel_frame->row(idx + 1);
		out_vel = (1 - phase) * vel0 + phase * vel1;
		
		//if(out_vel[1] > 100)
		//{
		//	printf("%lf\n%lf\n%lf\n%lf\n%lf\n%d\n\n\n", out_vel[1], time, vel0[1], vel1[1], phase, idx);sleep(0.1);
		//}
	}
}

void cMotion::CalcFramePhase(double phase, tFrame& out_frame, bool force_mirror /*=false*/) const
{
	double max_time = GetDuration();
	double time = phase * max_time;
	CalcFrame(time, out_frame, force_mirror);
}

bool cMotion::LoadJson(const Json::Value& root)
{
	bool succ = true;
	if (!root[gLoopKey].isNull())
	{
		std::string loop_str = root[gLoopKey].asString();
		ParseLoop(loop_str, mLoop);
	}

	mVelFilterCutoff = root.get(gVelFilterCutoffKey, mVelFilterCutoff).asDouble();

	if (mParams.mRightJoints.size() == 0 && mParams.mLeftJoints.size() == 0)
	{
		succ &= LoadJsonJoints(root, mParams.mRightJoints, mParams.mLeftJoints);
	}
	assert(mParams.mRightJoints.size() == mParams.mLeftJoints.size());

	if (!root[gFrameKey].isNull())
	{
		succ &= LoadJsonFrames(root[gFrameKey], mFrames);
	}

	return succ;
}

bool cMotion::ParseLoop(const std::string& str, eLoop& out_loop) const
{
	bool succ = false;
	for (int i = 0; i < eLoopMax; ++i)
	{
		if (str == gLoopStr[i])
		{
			out_loop = static_cast<eLoop>(i);
			succ = true;
			break;
		}
	}
	
	if (!succ)
	{
		printf("Unsupported loop mode: %s\n", str.c_str());
		assert(false);
		succ = false;
	}
	return succ;
}

bool cMotion::LoadJsonFrames(const Json::Value& root, Eigen::MatrixXd& out_frames) const
{
	bool succ = true;

	assert(root.isArray());
	int num_frames = root.size();

	int data_size = 0;
	if (num_frames > 0)
	{
		int idx0 = 0;
		Json::Value frame_json = root.get(idx0, 0);
		data_size = frame_json.size();
		out_frames.resize(num_frames, data_size);
	}

	for (int f = 0; f < num_frames; ++f)
	{
		Eigen::VectorXd curr_frame;
		succ &= ParseFrameJson(root.get(f, 0), curr_frame);
		if (succ)
		{
			assert(mFrames.cols() == curr_frame.size());
			out_frames.row(f) = curr_frame;
		}
		else
		{
			out_frames.resize(0, 0);
			break;
		}
	}
	return succ;
}

bool cMotion::ParseFrameJson(const Json::Value& root, Eigen::VectorXd& out_frame) const
{
	bool succ = false;
	if (root.isArray())
	{
		int data_size = root.size();
		out_frame.resize(data_size);
		for (int i = 0; i < data_size; ++i)
		{
			Json::Value json_elem = root.get(i, 0);
			out_frame[i] = json_elem.asDouble();
		}

		succ = true;
	}
	return succ;
}

bool cMotion::LoadJsonJoints(const Json::Value& root, std::vector<int>& out_right_joints, std::vector<int>& out_left_joints) const
{
	bool succ = true;
	if (!root[gRightJointsKey].isNull() && !root[gLeftJointsKey].isNull())
	{
		auto right_joints_json = root[gRightJointsKey];
		auto left_joints_json = root[gLeftJointsKey];
		assert(right_joints_json.isArray());
		assert(left_joints_json.isArray());

		int num_right_joints = right_joints_json.size();
		assert(num_right_joints == left_joints_json.size());
		succ = num_right_joints == left_joints_json.size();
		if (succ)
		{
			std::vector<int> right_joints(num_right_joints);
			std::vector<int> left_joints(num_right_joints);

			out_right_joints.resize(num_right_joints);
			out_left_joints.resize(num_right_joints);
			for (int i = 0; i < num_right_joints; ++i)
			{
				int right_id = right_joints_json[i].asInt();
				int left_id = left_joints_json[i].asInt();
				out_right_joints[i] = right_id;
				out_left_joints[i] = left_id;
			}
		}
	}
	else
	{
		out_right_joints.clear();
		out_left_joints.clear();
	}

	return succ;
}

std::string cMotion::BuildLoopStr(eLoop loop) const
{
	return gLoopStr[loop];
}

void cMotion::PostProcessFrames(Eigen::MatrixXd& frames) const
{
	int frame_size = GetFrameSize();
	int num_frames = static_cast<int>(frames.rows());
	double curr_time = gMinTime;

	for (int f = 0; f < num_frames; ++f)
	{
		auto curr_frame = frames.row(f);
		double duration = curr_frame(0, eFrameTime);
		curr_frame(0, eFrameTime) = curr_time;
		curr_time += duration;

		if (HasPostProcessFunc())
		{
			Eigen::VectorXd pose = curr_frame.segment(eFrameMax, frame_size - eFrameMax);
			mParams.mPostProcessFunc(&pose);
			curr_frame.segment(eFrameMax, frame_size - eFrameMax) = pose;
		}
	}
}

double cMotion::GetDuration() const
{
	int num_frames = GetNumFrames();
	double max_time = mFrames(num_frames - 1, eFrameTime);
	return max_time;
}

void cMotion::SetDuration(double dur)
{
	double dur_old = GetDuration();
	int num_frames = GetNumFrames();

	for (int f = 0; f < num_frames; ++f)
	{
		double t = GetFrameTime(f);
		t /= dur_old;
		t *= dur;
		SetFrameTime(f, t);
	}
	UpdateVel();
}

double cMotion::GetFrameTime(int f) const
{
	return mFrames(f, eFrameTime);
}

double cMotion::GetFrameDuration(int f) const
{
	double dur = 0;
	if (f < GetNumFrames() - 1)
	{
		dur = GetFrameTime(f + 1) - GetFrameTime(f);
	}
	return dur;
}

int cMotion::CalcCycleCount(double time) const
{
	double dur = GetDuration();
	double phases = time / dur;
	int count = static_cast<int>(std::floor(phases));
	bool loop = EnableLoop();
	count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
	return count;
}

void cMotion::CalcIndexPhase(double time, int& out_idx, double& out_phase) const
{
	double max_time = GetDuration();

	if (!EnableLoop())
	{
		if (time <= gMinTime)
		{
			out_idx = 0;
			out_phase = 0;
			return;
		}
		else if (time >= max_time)
		{
			out_idx = GetNumFrames() - 2;
			out_phase = 1;
			return;
		}
	}

	int cycle_count = CalcCycleCount(time);
	time -= cycle_count * GetDuration();
	if (time < 0)
	{
		time += max_time;
	}

	const Eigen::VectorXd& frame_times = mFrames.col(eFrameTime);
	auto it = std::upper_bound(frame_times.data(), frame_times.data() + frame_times.size(), time);
	out_idx = static_cast<int>(it - frame_times.data() - 1);

	double time0 = frame_times(out_idx);
	double time1 = frame_times(out_idx + 1);
	out_phase = (time - time0) / (time1 - time0);
}

void cMotion::UpdateVel()
{
	BuildFrameVel(mFrameVel, false);

	if (mLoop == eLoopMirror)
	{
		BuildFrameVel(mFrameVelMirror, true);
	}
}

bool cMotion::IsOver(double time) const
{
	return !EnableLoop() && (time >= GetDuration());
}

bool cMotion::HasBlendFunc() const
{
	return mParams.mBlendFunc != nullptr;
}

bool cMotion::HasMirrorFunc() const
{
	return mParams.mMirrorFunc != nullptr;
}

bool cMotion::HasVelFunc() const
{
	return mParams.mVelFunc != nullptr;
}

bool cMotion::HasPostProcessFunc() const
{
	return mParams.mPostProcessFunc != nullptr;
}

bool cMotion::EnableLoop() const
{
	return mLoop != eLoopNone;
}

cMotion::eLoop cMotion::GetLoop() const
{
	return mLoop;
}

void cMotion::SetLoop(eLoop loop)
{
	mLoop = loop;
}

void cMotion::BlendFramesIntern(const Eigen::VectorXd* a, const Eigen::VectorXd* b, double lerp, cMotion::tFrame* out_frame) const
{
	*out_frame = (1 - lerp) * (*a) + lerp * (*b);
}

bool cMotion::NeedMirrorFrame(double time) const
{
	bool mirror = false;
	if (mLoop == eLoopMirror)
	{
		int cycle = CalcCycleCount(time);
		mirror = cycle % 2 != 0;
	}
	return mirror;
}

void cMotion::MirrorFrame(tFrame& out_frame) const
{
	assert(mParams.mMirrorFunc != nullptr);
	assert(mParams.mLeftJoints.size() > 0 && mParams.mRightJoints.size() > 0);
	mParams.mMirrorFunc(&mParams.mLeftJoints, &mParams.mRightJoints, &out_frame);
}

void cMotion::CalcFrameVel(const tFrame& frame0, const tFrame& frame1, double dt, tFrame& out_vel) const
{
	if (HasVelFunc())
	{
		mParams.mVelFunc(&frame0, &frame1, dt, &out_vel);
	}
	else
	{
		out_vel = (frame1 - frame0) / dt;
	}
}

void cMotion::Output(const std::string& out_filepath) const
{
	FILE* file = cFileUtil::OpenFile(out_filepath, "w");

	fprintf(file, "{\n");
	fprintf(file, "\"%s\": ", gLoopKey.c_str());

	std::string loop_str = BuildLoopStr(mLoop);
	fprintf(file, "\"%s\",\n", loop_str.c_str());

	fprintf(file, "\"%s\": %.5f,\n", gVelFilterCutoffKey.c_str(), mVelFilterCutoff);

	if (mParams.mRightJoints.size() > 0 && mParams.mLeftJoints.size() > 0)
	{
		fprintf(file, "\"%s\": [", gRightJointsKey.c_str());
		for (size_t i = 0; i < mParams.mRightJoints.size(); ++i)
		{
			if (i != 0)
			{
				fprintf(file, ", ");
			}
			fprintf(file, "%i", mParams.mRightJoints[i]);
		}
		fprintf(file, "],\n");

		fprintf(file, "\"%s\": [", gLeftJointsKey.c_str());
		for (size_t i = 0; i < mParams.mLeftJoints.size(); ++i)
		{
			if (i != 0)
			{
				fprintf(file, ", ");
			}
			fprintf(file, "%i", mParams.mLeftJoints[i]);
		}
		fprintf(file, "],\n");
	}

	fprintf(file, "\"Frames\":\n[\n");

	int num_frames = GetNumFrames();
	for (int f = 0; f < num_frames; ++f)
	{
		if (f != 0)
		{
			fprintf(file, ",\n");
		}

		Eigen::VectorXd curr_frame = mFrames.row(f);
		double dur = 0;
		if (f < num_frames - 1)
		{
			dur = GetFrameDuration(f);
		}
		curr_frame[eFrameTime] = dur;
		std::string frame_json = cJsonUtil::BuildVectorJson(curr_frame);
		fprintf(file, "%s", frame_json.c_str());
	}

	fprintf(file, "\n]");
	fprintf(file, "\n}");
	cFileUtil::CloseFile(file);
}

void cMotion::Warp(Eigen::MatrixXd& out_frames, std::vector<tVector> Center, std::vector<tVector> GoalPos, std::vector<double> PertitionTime)
{

	int FrameNum = GetNumFrames();
	double frame_duration = GetFrameTime(1);

	double pertition_warped = 0.0;
	double pertition = 0.0;
	std::vector<double> PertitionTime_warped = {};
	for (int i = 0; i < PertitionTime.size(); i++)
	{
		double factor_tmp = (GoalPos[i][3] - pertition) / (Center[i][3] - pertition);
		if (factor_tmp < 0.5){
			factor_tmp = 0.5;
		}
		pertition_warped = pertition_warped + (PertitionTime[i] - pertition) * factor_tmp;
		pertition = PertitionTime[i];
		PertitionTime_warped.push_back(pertition_warped);
	}

	double dur_warped = PertitionTime_warped.back();
	int FrameNum_warped = int(FrameNum * dur_warped / GetDuration()) +1;
	Eigen::MatrixXd warped_frames(FrameNum_warped, GetFrameSize());

	double curr_time = 0;
	for(int f=0; f<FrameNum_warped; f++)
	{
		auto curr_frame = warped_frames.row(f);
		
		double phase = TracePhase(curr_time, PertitionTime, PertitionTime_warped);
		Eigen::VectorXd interp_pose;
		CalcFramePhase(phase, interp_pose);
		
		curr_frame(0, eFrameTime) = curr_time;
		curr_frame.tail(GetFrameSize()-1) = interp_pose;
		
		curr_time += frame_duration;
	}

	out_frames = warped_frames;

	/*
	int frame_size = GetFrameSize();
	int num_frames = static_cast<int>(out_frames.rows());
	double set_goal_duration = 0;
	double frame_duration = out_frames.row(0)(0, eFrameTime);

	for (int f = 0; f < keyframe_num; ++f)
	{
		auto curr_frame = out_frames.row(f);
		set_goal_duration += frame_duration;
	}

	//cannot use GetDuration(), use basic function BlendFrames() insted of CalcFrame()
	//modify motion in the same manner independently of Goaltime
	double factor = GoalTime / set_goal_duration;
	int out_frame_num = int(factor * out_frames.rows())+1;
	Eigen::MatrixXd expanded_frames(out_frame_num, out_frames.cols());
	for (int f = 0; f<out_frame_num; ++f)
	{
		double phase = double(f) / double(out_frame_num);
		int idx = int(phase * GetNumFrames());
		double lerp = phase * GetNumFrames() - idx;
		Eigen::VectorXd interp_pose;
		std::cout << f << " " << idx << " " << out_frame_num << " " << lerp <<  " " << phase << std::endl;
		BlendFrames(idx, idx+1, lerp, interp_pose);
		auto curr_frame = expanded_frames.row(f);
		curr_frame(0, eFrameTime) = frame_duration;
		curr_frame.tail(expanded_frames.cols()-1) = interp_pose;
		//std::cout << interp_pose << "  " << std::endl;
		//std::cout << out_frames.row(f) << "  " << std::endl;
		//std::cout << f << "  " << phase <<  "  " << curr_frame << std::endl;sleep(0.1);
	}
	//std::cout << extended_frames << std::endl;
	out_frames = expanded_frames;


	
	double factor = GoalTime / curr_time;
	for (int f=0; f < keyframe_num; ++f)
	{
		auto curr_frame = out_frames.row(f);
		double duration = curr_frame(0, eFrameTime);
		duration *= factor;
		curr_frame(0, eFrameTime) = duration;
		//printf("%f\n%f\n%f\n\n\n\n\n", GoalTime, factor, curr_time);sleep(0.1);
	}
	
	
	
	if (curr_time > GoalTime)
	{	
		double factor = GoalTime / curr_time;
		for (int f=0; f < keyframe_num; ++f)
		{
			auto curr_frame = out_frames.row(f);
			double duration = curr_frame(0, eFrameTime);
			duration *= factor;
			curr_frame(0, eFrameTime) = duration;
			//printf("%f\n%f\n%f\n\n\n\n\n", GoalTime, factor, curr_time);sleep(0.1);
		}
	}
	
	
	//if (curr_time < GoalTime)
	//{
		


		//stopped motion won't be useful
		
		auto first_frame = out_frames.row(0);
		double duration = first_frame(0, eFrameTime);
		int pad_frame_num = int((GoalTime - curr_time)/duration)+1;
		Eigen::MatrixXd pad(pad_frame_num, out_frames.cols());
		for (int f=0; f<pad_frame_num; ++f)
		{
			auto curr_pad_frame = pad.row(f);
			curr_pad_frame = first_frame;
		}
		Eigen::MatrixXd concat_frames(pad.rows()+out_frames.rows(), out_frames.cols());
		concat_frames << pad, out_frames;
		out_frames = concat_frames;
		

	    //very long frame will corrupt
		//auto curr_frame = out_frames.row(0);
		//double duration = curr_frame(0, eFrameTime);
		//curr_frame(0, eFrameTime) += GoalTime - curr_time;
		//printf("%f\n%f\n%d\n\n\n\n\n", GoalTime, curr_time, keyframe_num);sleep(0.1);
	//}
	*/
	

}

double cMotion::TracePhase(double time_warped, std::vector<double> pertition_time, std::vector<double> pertition_time_warped)
{
	double phase = 0.0;

	int idx = ChoosePertitionIdx(time_warped, pertition_time_warped);

	std::vector<double> pertition_time_pad = {0.0};
	std::vector<double> pertition_time_warped_pad = {0.0};
	pertition_time_pad.insert(pertition_time_pad.end(), pertition_time.begin(), pertition_time.end());
	pertition_time_warped_pad.insert(pertition_time_warped_pad.end(), pertition_time_warped.begin(), pertition_time_warped.end());

	std::vector<double> pertition_phase_pad = {0.0};
	for (int i = 0; i < pertition_time.size(); i++){
		pertition_phase_pad.push_back(pertition_time[i] / pertition_time.back());
	}

	double factor = (pertition_time_pad[idx + 1] - pertition_time_pad[idx]) / (pertition_time_warped_pad[idx + 1] - pertition_time_warped_pad[idx]);
	phase += pertition_phase_pad[idx];
	phase += (time_warped - pertition_time_warped_pad[idx]) * factor / pertition_time_pad.back();
	
	return phase;
}


void cMotion::FabrikIK(Eigen::MatrixXd& out_frames, std::vector<tVector> GoalPos, std::vector<tVector> Center, std::vector<int> IKjoint, std::vector<double> PertitionTime, double WindowTime) {
	
	JavaVM *jvm;
	JNIEnv *env;

	JNI_GetCreatedJavaVMs(&jvm, 1, NULL);
	
	int res = jvm->AttachCurrentThread((void **)&env, NULL);
	if (res < 0){
		std::cerr << "Java VM attach faild:" << res << std::endl;
		return;
	}

	// クラス検索
    jclass cls = env->FindClass("au/edu/federation/caliko/demo3d/Test");
    if(cls == 0){
        std::cout << "could not found class : Test" << std::endl;
        return;
    }
 
    // クラス情報をインスタンス化するために<init>メソッドのメソッドIDを取得
    jmethodID cns = env->GetMethodID(cls, "<init>", "()V");
    if(cns == NULL){
        std::cout << "could not get <init> method." << std::endl;
        return;
    }
    jobject obj = env->NewObject(cls, cns);

	// 各メソッドをインスタンスから呼び出す
    //  第3引数のシグネチャは javap -s Test などで調べられる(引数と返値を表す)
    jmethodID mid = env->GetMethodID(cls, "run", "([D[DI)[D");
    if(mid == NULL){
        std::cout << "could not get method : " << "run" << std::endl;
        return;
    }

	int frame_num = mFrames.rows();

	Eigen::MatrixXd modified_frames(frame_num, GetFrameSize());
	for (int i=0; i < frame_num; i++){
		double t = GetFrameTime(i);
		double dur = GetDuration();
		int idx = ChoosePertitionIdx(t, PertitionTime);
		tVector shift = GoalPos[idx] - Center[idx];
		double T = Center[idx][3];
		int ikjoint = IKjoint[idx];
		Eigen::VectorXd frame = mFrames.row(i);
		
		double factor;
		/*
		if (t > T){
			factor = (dur - t) / (dur - T);
		} else {
			factor = t/T;
		}
		*/
		double window = WindowTime;
		double abs_diff = abs(T - t);
		if (abs_diff < window){
			factor = 1 - abs_diff / window;
		} else {
			factor = 0.0;
		}
		
		tVector shift_i = shift * factor;// * factor;
		
		std::vector<double> vec_frame = EigenVec2Vec(frame);
		int size_frame = GetFrameSize();
		jdoubleArray framej = (jdoubleArray) env->NewDoubleArray(size_frame);
		env->SetDoubleArrayRegion(framej, 0, size_frame, &vec_frame[0]);

		std::vector<double> vec_shift = EigenVec2Vec(shift_i);
		int size_goal = (shift_i).size();
		jdoubleArray shiftj = (jdoubleArray) env->NewDoubleArray(size_goal);
		env->SetDoubleArrayRegion(shiftj, 0, size_goal, &vec_shift[0]);
		
		jint endeffj = ikjoint;
		
		jdoubleArray resultj = (jdoubleArray)(env->CallObjectMethod(obj, mid, framej, shiftj, endeffj));
		jsize sizej_result = env->GetArrayLength(resultj);
		std::vector<double> result(sizej_result);
		env->GetDoubleArrayRegion(resultj, 0, sizej_result, &result[0]);
	
		auto curr_frame = modified_frames.row(i);
		curr_frame = Eigen::VectorXd::Map(result.data(), result.size());
	}
	out_frames = modified_frames;

	/*
	// jstring -> char*の変換
    const char* cstr = env->GetStringUTFChars(jstr, 0);
    char *str = strdup(cstr);
    env->ReleaseStringUTFChars(jstr, cstr);
    std::cout << str << std::endl;
	*/

	res = jvm->DetachCurrentThread();
	if (res < 0){
		std::cerr << "Java VM detach faild:" << res << std::endl;
		return;
	}
	
}

int cMotion::ChoosePertitionIdx(double time, std::vector<double> PertitionTime) const
{
	for (int i; i < PertitionTime.size(); i++)
	{
		if (time > PertitionTime[i] && i < PertitionTime.size() - 1){
			continue;
		} else {
			return i;
		}
	}
}

std::vector<double> cMotion::EigenVec2Vec(Eigen::VectorXd vec) {
	std::vector<double> stdvec;
	stdvec.resize(vec.size());
	Eigen::VectorXd::Map(&stdvec[0], vec.size()) = vec;
	return stdvec;
}