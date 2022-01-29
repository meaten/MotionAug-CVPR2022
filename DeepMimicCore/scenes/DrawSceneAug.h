#pragma once

#include "scenes/DrawRLScene.h"
#include "scenes/DrawSceneSimChar.h"
#include "anim/KinCharacter.h"

class cDrawSceneAug : virtual public cDrawRLScene, virtual public cDrawSceneSimChar
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	cDrawSceneAug();
	virtual ~cDrawSceneAug();

	virtual void Init();
	virtual void Clear();
	virtual bool IsEpisodeEnd() const;
	virtual bool CheckValidEpisode() const;

	virtual void Keyboard(unsigned char key, double device_x, double device_y);
	virtual void DrawKinChar(bool enable);

	virtual void DrawMisc() const;
	virtual void DrawGoals() const;
	virtual void DrawGoal(const tVector goalpos, int flag) const;
	virtual void DrawGoalShape() const;
	virtual void DrawGoalEllipsoids() const;
	virtual void DrawGoalEllipsoid() const;
	virtual void DrawGoalFanShapes() const;
	virtual void DrawGoalFanShape() const;
	virtual std::string GetName() const;
	virtual void DrawPoliInfo() const;
protected:

	bool mDrawKinChar;

	virtual cRLScene* GetRLScene() const;

	virtual void BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const;
	virtual void DrawCharacters() const;
	virtual void DrawKinCharacters() const;
	virtual void DrawKinCharacter(const std::shared_ptr<cKinCharacter>& kin_char) const;
	
	virtual const std::shared_ptr<cKinCharacter>& GetKinChar() const;
};
