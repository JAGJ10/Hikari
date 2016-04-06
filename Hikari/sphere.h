#ifndef SPHERE_H
#define SPHERE_H

#include "ray.h"
#include "helper_math.h"
enum Refl_t { DIFF, SPEC, REFR };
struct Sphere {
	float radius;
	float3 position, emission, color;
	Refl_t refl;

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit
		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = position - r.origin;  // 
		float t, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + radius*radius; // discriminant
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
	}
};

#endif