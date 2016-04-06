#ifndef AABB_H
#define AABB_H

#include <cuda_runtime.h>
#include "ray.h"
#include <vector>

class AABB {
public:
	float3 minBounds, maxBounds;

	__host__ __device__ AABB() {
		minBounds = make_float3(FLT_MAX);
		maxBounds = make_float3(-FLT_MAX);
	}

	__host__ __device__ AABB(const float3& min, const float3& max) {
		this->minBounds = min;
		this->maxBounds = max;
	}

	__host__ __device__ bool intersect(const Ray &r) const {
		float tmin = -FLT_MAX;
		float tmax = FLT_MAX;
		float3 t0 = (minBounds - r.origin) * r.invDir;
		float3 t1 = (maxBounds - r.origin) * r.invDir;

		if (r.invDir.x < 0.0f) { float tmp = t0.x; t0.x = t1.x; t1.x = tmp; }
		tmin = t0.x > tmin ? t0.x : tmin;
		tmax = t1.x < tmax ? t1.x : tmax;
		if (tmax < tmin) return false;

		if (r.invDir.y < 0.0f) { float tmp = t0.y; t0.y = t1.y; t1.y = tmp; }
		tmin = t0.y > tmin ? t0.y : tmin;
		tmax = t1.y < tmax ? t1.y : tmax;
		if (tmax < tmin) return false;

		if (r.invDir.z < 0.0f) { float tmp = t0.z; t0.z = t1.z; t1.z = tmp; }
		tmin = t0.z > tmin ? t0.z : tmin;
		tmax = t1.z < tmax ? t1.z : tmax;
		if (tmax < tmin) return false;

		return true;
	}

	__host__ __device__ float3 getCentroid() const {
		return (minBounds + maxBounds) * 0.5f;
	}
};

#endif