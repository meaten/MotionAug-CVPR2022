#pragma once

#include <stdlib.h>

#include "DrawSceneAugRes.h"
#include "SceneAugRes.h"
#include "render/DrawCharacter.h"
#include "render/DrawSimCharacter.h"
#include "render/DrawUtil.h"
#include "sim/RBDUtil.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>

const double gLinkWidth = 0.025f;
const tVector gLineColor = tVector(0, 0, 0, 1);
const tVector gFilLColor = tVector(0.6f, 0.65f, 0.675f, 1);

void cDrawSceneAugRes::DrawMisc() const
{
	cDrawSceneAug::DrawMisc();
    DrawResForce();
}

void cDrawSceneAugRes::BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const
{
	out_scene = std::shared_ptr<cSceneAugRes>(new cSceneAugRes());
}

const std::shared_ptr<cKinCharacter>& cDrawSceneAugRes::GetKinChar() const
{
	const cSceneAug* scene = dynamic_cast<const cSceneAug*>(mScene.get());
	return scene->GetKinChar();
}

void cDrawSceneAugRes::DrawPoliInfo() const
{
	cDrawSceneAug::DrawPoliInfo();
}


void cDrawSceneAugRes::DrawResForce() const
{
    const auto& sim_char = mScene->GetCharacter(0);
    const auto ctrl = sim_char->GetController();    
    const auto& ctrl_res = dynamic_cast<cCtPDResController*>(ctrl.get());

    Eigen::VectorXd ResForce(6);
    ResForce = ctrl_res->GetResForce();

    if (ResForce.rows() != 0){

        tVector force =  tVector(ResForce[0], ResForce[1], ResForce[2], 0);
	    tVector torque = tVector(ResForce[3], ResForce[4], ResForce[5], 0);

        tVector pos = sim_char->CalcJointPos(0);
        pos[2] += 0.2;

        cDrawPerturb::DrawForce(pos, force);
        cDrawPerturb::DrawTorque(pos, torque);
    }
}

cDrawSceneAugRes::cDrawSceneAugRes() : cDrawSceneAug()
{
}

cDrawSceneAugRes::~cDrawSceneAugRes()
{
}

