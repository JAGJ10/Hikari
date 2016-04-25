#ifndef AREALIGHT_H
#define AREALIGHT_H

#include <cuda_runtime.h>

class AreaLight {
public:
	AreaLight(float3 pos, float width, float height, float3 color);
	~AreaLight();

	float3 sample(float r1, float r2);

private:
	float3 pos, color;
	float width, height;
};

#endif