#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>
#include "helper_math.h"

struct HitInfo {
	int tri;
	float hitDist;
	float3 hitPoint;

	__host__ __device__ HitInfo() : tri(-1), hitDist(1e20f) {}
};

struct Ray {
	float3 origin;
	float3 dir;
	float3 invDir;

	__host__ __device__ Ray() {}
	__host__ __device__ Ray(float3 o, float3 d) : origin(o), dir(d), invDir(1.0f / d) {}
};

#endif