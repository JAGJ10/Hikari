#include "AreaLight.h"
#include "helper_math.h"

AreaLight::AreaLight(float3 pos, float width, float height, float3 color) : pos(pos), width(width), height(height), color(color) {}

AreaLight::~AreaLight() {}

float3 AreaLight::sample(float r1, float r2) {
	return make_float3(0);
}