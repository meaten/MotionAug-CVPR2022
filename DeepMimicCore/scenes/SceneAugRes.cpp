#pragma once

#include "SceneAugRes.h"
#include "sim/RBDUtil.h"
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

void cSceneAugRes::CalcReward(const cSimCharacter& sim_char, const cKinCharacter& ref_char, double &r, double &r_i, double &r_s) const
{
	double omega_goal = 3.0 * mEnableStrikeReward;
	double omega_imitate = 0.9;
	double omega_resforce = 0.1;
	double omega_total = omega_goal + omega_imitate + omega_resforce;
	omega_goal /= omega_total;
	omega_imitate /= omega_total;
	omega_resforce /= omega_total;

	double r_imitate = CalcRewardImitate(sim_char, ref_char);
	double r_goal = 0.0;
	if (mCenterT.size() > 0) {
		r_goal = CalcRewardGoal(sim_char, ref_char);
	}
	double r_resforce = CalcRewardResForce(sim_char);
	double reward = omega_imitate * r_imitate + omega_goal * r_goal + omega_resforce * r_resforce;
	r = reward; r_i = r_imitate; r_s = r_goal;
	//std::cerr << reward << std::endl << r_imitate << std::endl << r_goal << std::endl << r_resforce << std::endl << std::endl;
}

double cSceneAugRes::CalcRewardResForce(const cSimCharacter& sim_char) const
{
	double resforce_w = 0.01;

	Eigen::VectorXd resforce = dynamic_cast<cCtPDResController*>(sim_char.GetController().get())->GetResForce();
	double norm_square = resforce.norm();
	
	return exp(- resforce_w * norm_square);
}

cSceneAugRes::cSceneAugRes() : cSceneAug()
{
}

cSceneAugRes::~cSceneAugRes()
{
}