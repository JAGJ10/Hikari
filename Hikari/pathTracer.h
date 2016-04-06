#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include "Camera.hpp"
#include "sphere.h"
#include "Ray.h"

class PathTracer {
public:
	PathTracer(Camera* cam);
	~PathTracer();

private:
	void initScene();
};

#endif