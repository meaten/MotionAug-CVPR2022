#pragma once

#include <stdlib.h>

#include "DrawSceneAug.h"
#include "SceneAug.h"
#include "render/DrawCharacter.h"
#include "render/DrawSimCharacter.h"
#include "render/DrawUtil.h"
#include "sim/RBDUtil.h"

#include "sim/CtController.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

const double gLinkWidth = 0.025f;
const tVector gLineColor = tVector(0, 0, 0, 1);
const tVector gFilLColor = tVector(0.6f, 0.65f, 0.675f, 1);

cDrawSceneAug::cDrawSceneAug()
{
	mDrawKinChar = false;
}

cDrawSceneAug::~cDrawSceneAug()
{
}

void cDrawSceneAug::Init()
{
	cDrawSceneSimChar::Init();
	cRLScene::Init();
}

void cDrawSceneAug::Clear()
{
	cDrawSceneSimChar::Clear();
	cDrawRLScene::Clear();
}

bool cDrawSceneAug::IsEpisodeEnd() const
{
	return cDrawRLScene::IsEpisodeEnd();
}

bool cDrawSceneAug::CheckValidEpisode() const
{
	return cDrawRLScene::CheckValidEpisode();
}

void cDrawSceneAug::Keyboard(unsigned char key, double device_x, double device_y)
{
	cDrawSceneSimChar::Keyboard(key, device_x, device_y);

	switch (key)
	{
	case 'k':
		DrawKinChar(!mDrawKinChar);
		break;
	default:
		break;
	}
}

void cDrawSceneAug::DrawKinChar(bool enable)
{
	mDrawKinChar = enable;
	if (mDrawKinChar)
	{
		printf("Enabled draw kinematic character\n");
	}
	else
	{
		printf("Disabled draw kinematic character\n");
	}
}

void cDrawSceneAug::DrawMisc() const
{
	
	if (mEnableTrace)
	{
		DrawTrace();
	}
	DrawPerturbs();
	const auto& curr_char = mScene->GetCharacter(0);
	const auto& curr_ctrl = curr_char->GetController();
	auto ctrl = dynamic_cast<cCtController*>(curr_ctrl.get());
	if (ctrl->GetGoalSize() > 0){
		DrawGoals();
		DrawGoalShape();
	}
}

void cDrawSceneAug::DrawGoals() const
{
	int num_chars = mScene->GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		const auto& curr_char = mScene->GetCharacter(i);
		const auto& curr_ctrl = curr_char->GetController();
		auto ctrl = dynamic_cast<cCtController*>(curr_ctrl.get());
		tVector goalpos = ctrl->GetGoalPos();
		int flag = ctrl->GetGoalFlagTime();
		DrawGoal(goalpos, flag);
	}
}

void cDrawSceneAug::DrawGoal(tVector goalpos, int flag) const
{
	tVector size = Eigen::VectorXd::Ones(4);
	size *= 100;
	cDrawUtil::PushMatrixView();
	//cDrawUtil::MultMatrixView(parent_world_trans);
	//cDrawUtil::Translate(Eigen::VectorXd::Ones(4));
	cDrawUtil::Translate(goalpos);
	//cDrawUtil::Rotate(theta, axis);
	double col = double(flag);
	//double col = 0.0;
	//printf("%lf\n", col);
	cDrawUtil::SetColor(tVector(col,col,col,0.5));
	cDrawUtil::DrawSphere(0.1, cDrawUtil::eDrawSolid);
	cDrawUtil::PopMatrixView();
}

void cDrawSceneAug::DrawGoalShape() const
{
	const auto& curr_char = mScene->GetCharacter(0);
	const auto& curr_ctrl = dynamic_cast<cCtController*>(curr_char->GetController().get());
	std::string goal_shape = curr_ctrl->GetGoalShape();
	if(goal_shape=="Ellipsoid"){
		DrawGoalEllipsoids();
	}else if(goal_shape=="FanShape"){
		DrawGoalFanShapes();
	}else if(goal_shape=="Fixed"){
		
	}else{
		
	}
}

void cDrawSceneAug::DrawGoalEllipsoids() const
{
	int num_chars = mScene->GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		DrawGoalEllipsoid();
	}
}

void cDrawSceneAug::DrawGoalEllipsoid() const
{
	cDrawUtil::eDrawMode draw_mode = cDrawUtil::eDrawWire;

	const auto& curr_char = mScene->GetCharacter(0);
	const auto& ctrl = dynamic_cast<cCtController*>(curr_char->GetController().get());
	tVector center = ctrl->GetGoalCenter(ctrl->GetTime());
	tVector minus  = ctrl->GetGoalEllipsoidMinus();
	tVector plus   = ctrl->GetGoalEllipsoidPlus();
	cDrawUtil::PushMatrixView();
	cDrawUtil::Translate(center);
	cDrawUtil::SetColor(tVector(0,0,0,0.5));

	cDrawUtil::Drawoctosphere(plus[0], plus[1], minus[2], draw_mode);
	cDrawUtil::Rotate(0.5 * M_PI, tVector(0,1,0,0));
	cDrawUtil::Drawoctosphere(minus[2], plus[1], minus[0], draw_mode);
	cDrawUtil::Rotate(0.5 * M_PI, tVector(0,1,0,0));
	cDrawUtil::Drawoctosphere(minus[0], plus[1], plus[2], draw_mode);
	cDrawUtil::Rotate(0.5 * M_PI, tVector(0,1,0,0));
	cDrawUtil::Drawoctosphere(plus[2], plus[1], plus[0], draw_mode);
	cDrawUtil::Rotate(M_PI, tVector(0,0,1,0));
	cDrawUtil::Drawoctosphere(minus[2], minus[1], plus[0], draw_mode);
	cDrawUtil::Rotate( 0.5 * M_PI, tVector(0,1,0,0));
	cDrawUtil::Drawoctosphere(plus[0], minus[1], plus[2], draw_mode);
	cDrawUtil::Rotate( 0.5 * M_PI, tVector(0,1,0,0));
	cDrawUtil::Drawoctosphere(plus[2], minus[1], minus[0], draw_mode);
	cDrawUtil::Rotate( 0.5 * M_PI, tVector(0,1,0,0));
	cDrawUtil::Drawoctosphere(minus[0], minus[1], minus[2], draw_mode);
	cDrawUtil::PopMatrixView();
}

void cDrawSceneAug::DrawGoalFanShapes() const
{
	int num_chars = mScene->GetNumChars();
	for (int i=0; i < num_chars; ++i)
	{
		DrawGoalFanShape();
	}
}

void cDrawSceneAug::DrawGoalFanShape() const
{
	const int grid = 20;

	cDrawUtil::eDrawMode draw_mode = cDrawUtil::eDrawSolid;

	const auto& curr_char = mScene->GetCharacter(0);
	const auto& ctrl = dynamic_cast<cCtController*>(curr_char->GetController().get());
	tVector high = ctrl->GetGoalFanShapeHigh();
	tVector low  = ctrl->GetGoalFanShapeLow();

	double width = 0.05;
	//r direction
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], high[1], high[2]), ctrl->CalcGoalPosFanShape(low[0], high[1], high[2]), width, draw_mode);
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], low[1],  high[2]), ctrl->CalcGoalPosFanShape(low[0], low[1],  high[2]), width, draw_mode);
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], high[1], low[2]), ctrl->CalcGoalPosFanShape(low[0],  high[1], low[2]), width, draw_mode);
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], low[1],  low[2]), ctrl->CalcGoalPosFanShape(low[0],  low[1],  low[2]), width, draw_mode);
	
	//h direction
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], high[1], high[2]), ctrl->CalcGoalPosFanShape(high[0], low[1], high[2]), width, draw_mode);
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(low[0],  high[1], high[2]), ctrl->CalcGoalPosFanShape(low[0],  low[1], high[2]), width, draw_mode);
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], high[1], low[2]), ctrl->CalcGoalPosFanShape(high[0], low[1],  low[2]), width, draw_mode);
	cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(low[0],  high[1], low[2]), ctrl->CalcGoalPosFanShape(low[0],  low[1],  low[2]), width, draw_mode);

	//phi direction
	double delta_phi = (high[2] - low[2]) / grid;
	for (int i=0; i<grid; ++i)
	{
		cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], high[1], low[2]+(i+1)*delta_phi), ctrl->CalcGoalPosFanShape(high[0], high[1], low[2]+i*delta_phi), width, draw_mode);
		cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(high[0], low[1], low[2]+(i+1)*delta_phi), ctrl->CalcGoalPosFanShape(high[0], low[1], low[2]+i*delta_phi), width, draw_mode);
		cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(low[0], high[1], low[2]+(i+1)*delta_phi), ctrl->CalcGoalPosFanShape(low[0], high[1], low[2]+i*delta_phi), width, draw_mode);
		cDrawUtil::DrawStrip(ctrl->CalcGoalPosFanShape(low[0], low[1], low[2]+(i+1)*delta_phi), ctrl->CalcGoalPosFanShape(low[0], low[1], low[2]+i*delta_phi), width, draw_mode);
	}
}

std::string cDrawSceneAug::GetName() const
{
	return cDrawRLScene::GetName();
}

cRLScene* cDrawSceneAug::GetRLScene() const
{
	return dynamic_cast<cRLScene*>(mScene.get());
}

void cDrawSceneAug::BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const
{
	out_scene = std::shared_ptr<cSceneAug>(new cSceneAug());
}

void cDrawSceneAug::DrawCharacters() const
{
	if (mDrawKinChar)
	{
		DrawKinCharacters();
	}
	cDrawSceneSimChar::DrawCharacters();
}

void cDrawSceneAug::DrawKinCharacters() const
{
	const auto& kin_char = GetKinChar();
	DrawKinCharacter(kin_char);
}
void cDrawSceneAug::DrawKinCharacter(const std::shared_ptr<cKinCharacter>& kin_char) const
{
	cDrawCharacter::Draw(*kin_char, gLinkWidth, gFilLColor, gLineColor);
}

const std::shared_ptr<cKinCharacter>& cDrawSceneAug::GetKinChar() const
{
	const cSceneAug* scene = dynamic_cast<const cSceneAug*>(mScene.get());
	return scene->GetKinChar();
}


void cDrawSceneAug::DrawPoliInfo() const
{
	cDrawSceneSimChar::DrawPoliInfo();
	const cSceneAug* scene = dynamic_cast<const cSceneAug*>(mScene.get());
	cCircularBuffer<double> rlog, rlog_imitate, rlog_strike;
	scene->GetRewardLog(rlog, rlog_imitate, rlog_strike);
	cDrawSimCharacter::DrawInfoRewardLog(rlog, rlog_imitate, rlog_strike, mCamera, 0.5);
}
