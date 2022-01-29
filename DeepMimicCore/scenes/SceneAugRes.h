#pragma once

#include "scenes/SceneAug.h"
#include "sim/CtPDResController.h"

class cSceneAugRes : virtual public cSceneAug
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cSceneAugRes();
	virtual ~cSceneAugRes();
protected:
	virtual void CalcReward(const cSimCharacter& sim_char, const cKinCharacter& ref_char, double &r, double &r_i, double &r_s) const;
	virtual double CalcRewardResForce(const cSimCharacter& sim_char) const;
};
