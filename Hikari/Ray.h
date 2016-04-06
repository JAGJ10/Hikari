#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>
#include "helper_math.h"

struct Ray {
	float3 origin;
	float3 dir;
	float3 invDir;

	__host__ __device__ Ray() {}
	__host__ __device__ Ray(float3 o, float3 d) : origin(o), dir(d), invDir(1.0f / d) {}
};

#endif