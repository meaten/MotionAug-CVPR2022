#pragma once

#include "scenes/DrawSceneAug.h"
#include "render/DrawPerturb.h"

class cDrawSceneAugRes : virtual public cDrawSceneAug
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	cDrawSceneAugRes();
	virtual ~cDrawSceneAugRes();
	virtual void DrawPoliInfo() const;
protected:
	virtual void DrawMisc() const;
	virtual void DrawResForce() const;
	virtual void BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const;
	virtual const std::shared_ptr<cKinCharacter>& GetKinChar() const;
};
