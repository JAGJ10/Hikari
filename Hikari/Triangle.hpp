#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <device_launch_parameters.h>
#include "helper_math.h"
#include "AABB.hpp"

struct triBox {
	int index;
	AABB bb;

	triBox(int triIndex, float3 min, float3 max) : index(triIndex), bb(min, max) {}
};

class Triangle {
public:
	float4 v0, v1, v2;
	float3 normal;
	float3 diffuse, emit;

	Triangle(float4 v0, float4 v1, float4 v2, float3 normal, float3 diffuse, float3 emit) : 
		v0(v0), v1(v1), v2(v2), normal(normal), diffuse(diffuse), emit(emit) {}

	__device__ float intersect(const Ray& r, const float3& v0, const float3& edge1, const float3& edge2) {
		float3 tvec = r.origin - v0;
		float3 pvec = cross(r.dir, edge2);
		float  det = 1.0f / dot(edge1, pvec);

		//det = __fdividef(1.0f, det);  // CUDA intrinsic function 

		float u = dot(tvec, pvec) * det;

		if (u < 0.0f || u > 1.0f) return -1.0f;

		float3 qvec = cross(tvec, edge1);
		float v = dot(r.dir, qvec) * det;

		if (v < 0.0f || (u + v) > 1.0f) return -1.0f;

		return dot(edge2, qvec) * det;
	}
};

#endif