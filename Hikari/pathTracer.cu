#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_math.h"
#include "sphere.h"
#include "Ray.h"
#include "Camera.hpp"
#include "bvh.h"

#define M_PI                  3.1415926535897932384626422832795028841971f
#define TWO_PI				  6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD     0.5773502691896257645091487805019574556476f
#define E                     2.7182818284590452353602874713526624977572f
#define width 1280
#define height 720

__device__ Triangle* triangles;
__device__ LBVHNode* nodes;

__constant__ Sphere spheres[] = {
	// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
	{ 1e5f, { 0.f, 0.f, -1e5f - 15.0f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, SPEC }, //Back 
	{ 1e5f, { 0.f, -1e5f - 2.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Bottom 
	{ 400.0f, { 0.0f, 465.0f, 0.f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

__device__ bool intersect(const Ray& r, float& hitdist, int& triId, int& geomtype, float3& pointHitInWorldSpace, bool& isLight) {
	int stack[64];
	int topIndex = 0;
	stack[topIndex++] = 0;
	bool intersected = false;

	while (topIndex) {
		const LBVHNode* n = &nodes[stack[--topIndex]];
		if (n->bb.intersect(r)) {
			if (n->triCount > 0) {
				uint32_t index = n->triOffset;
				for (uint32_t i = 0; i < n->triCount; i++) {
					Triangle tri = triangles[index + i];
					float t = tri.intersect(r, make_float3(tri.v0), make_float3(tri.v1 - tri.v0), make_float3(tri.v2 - tri.v0));
					if (t < hitdist && t > 0.001f) {
						isLight = tri.isLight;
						hitdist = t;
						triId = index + i;
						geomtype = 2;
						pointHitInWorldSpace = r.origin + r.dir * t;
						intersected = true;
					}
				}
			} else {
				stack[topIndex++] = n->rightChild;
				stack[topIndex++] = (n - nodes) + 1;
			}
		}
	}

	return intersected;
}

__device__ float3 radiance(Ray& r, curandState* randState, const int numTriangles, int avoidSelf) {
	// colour mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);

	for (int bounces = 0; bounces < 5; bounces++) {
		float d = 1e20;
		float scene_t = 1e20;
		float3 pointHitInWorldSpace;

		// reset scene intersection function parameters
		int sphere_id = -1;
		int triangle_id = -1;
		int geomtype = -1;
		float3 f = make_float3(0);  // primitive colour
		float3 emit = make_float3(0); // primitive emission colour
		float3 x; // intersection point
		float3 n; // normal
		float3 nl; // oriented normal
		float3 dw; // ray dir of next path segment
		Refl_t refltype = REFR;

		bool isLight = false;
		intersect(r, scene_t, triangle_id, geomtype, pointHitInWorldSpace, isLight);

		// SPHERES
		// intersect all spheres in the scene
		float numspheres = sizeof(spheres) / sizeof(Sphere);
		for (int i = int(numspheres); i--;)  // for all spheres in scene
			// keep track of distance from origin to closest intersection point
			if ((d = spheres[i].intersect(r)) && d < scene_t) { scene_t = d; sphere_id = i; geomtype = 1; }

		//exit early
		if (geomtype == -1) return accucolor;

		// if sphere:
		if (geomtype == 1) {
			Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
			x = r.origin + r.dir*scene_t;  // intersection point on object
			n = normalize(x - sphere.position);		// normal
			nl = dot(n, r.dir) < 0 ? n : n * -1; // correctly oriented normal
			f = sphere.color;   // object colour
			refltype = sphere.refl;
			emit = sphere.emission;  // object emission
			accucolor += (mask * emit);
		} else if (geomtype == 2) { // if Triangle:
			x = pointHitInWorldSpace;  // intersection point
			n = triangles[triangle_id].normal;  // normal 
			nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal

			// colour, refltype and emit value are hardcoded and apply to all triangles
			// no per Triangle material support yet
			f = make_float3(1.0f);  // Triangle colour
			refltype = DIFF;
			emit = make_float3(0.0f, 0.0f, 0.0f);
			accucolor += (mask * emit);
		}

		// SHADING: diffuse, specular

		// ideal diffuse reflection (see "Realistic Ray Tracing", P. Shirley)
		if (refltype == DIFF) {
			// create 2 random numbers
			float r1 = 2 * M_PI * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float3 w = normalize(nl);
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			// compute cosine weighted random ray dir on hemisphere 
			dw = normalize(u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2));

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01f;

			// multiply mask with colour of object
			mask *= f;
		} else if (refltype == SPEC) { // ideal specular reflection (mirror) 
			// compute relfected ray dir according to Snell's law
			dw = r.dir - 2.0f * n * dot(n, r.dir);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + nl * 0.01;

			// multiply mask with colour of object
			mask *= f;
		}

		// set up origin and dir of next path segment
		r.origin = pointHitInWorldSpace;
		r.dir = dw;
		r.invDir = 1.0f / dw;
	}

	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return accucolor;
}

__global__ void pathTrace(Camera* cam, cudaSurfaceObject_t surface, float4* buffer, const int numTriangles, unsigned int frameNumber, unsigned int hashed) {
	//thrust::default_random_engine rng(randhash(hashed) * randhash(i));
	//thrust::uniform_real_distribution<float> uniformDistribution(0, 1);
	//float r1 = uniformDistribution(rng);
	//float r2 = uniformDistribution(rng);

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= (cam->res.x * cam->res.y)) return;

	curandState randState;
	curand_init(hashed + index, 0, 0, &randState);

	float r1 = curand_uniform(&randState);
	float r2 = curand_uniform(&randState);
	Ray ray = cam->getRay(x, y, true, r1, r2, frameNumber);
	float3 r = radiance(ray, &randState, numTriangles, -1);

	// write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
	buffer[index] += make_float4(r, 1.0f);
	float4 tempcol = buffer[index] / frameNumber;
	//tempcol = make_float4(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f), 1.0f);
	//tempcol /= 2.2f;
	surf2Dwrite(tempcol, surface, x * sizeof(float4), (height - y - 1));
}

__device__ bool shadowRay(float3 pointHitInWorldSpace, float3 lightPos) {
	float3 rayDir = normalize(lightPos - pointHitInWorldSpace);
	Ray shadowRay(pointHitInWorldSpace + 0.01f * rayDir, rayDir);
	int geomtype = -1;
	int triangle_id = -1;
	float scene_t = 1e20;
	float3 point;
	bool isLight;
	if (!intersect(shadowRay, scene_t, triangle_id, geomtype, point, isLight)) {
		return false;
	} else if (isLight) {
		return true;
	}
	
	return false;
}

__global__ void primaryRays(Camera* cam, cudaSurfaceObject_t surface, float4* buffer, unsigned int frameNumber, unsigned int hashed) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= (cam->res.x * cam->res.y)) return;

	curandState randState;
	curand_init(hashed + index, 0, 0, &randState);

	float r1 = curand_uniform(&randState);
	float r2 = curand_uniform(&randState);
	Ray r = cam->getRay(x, y, true, r1, r2, frameNumber);

	//Intersect primary ray with scene and then do light sampling
	float d = 1e20;
	float scene_t = 1e20;
	float3 pointHitInWorldSpace;

	// reset scene intersection function parameters
	int sphere_id = -1;
	int triangle_id = -1;
	int geomtype = -1;
	float3 f = make_float3(0);  // primitive colour
	float3 emit = make_float3(0); // primitive emission colour
	float3 x1; // intersection point
	float3 n; // normal
	float3 nl; // oriented normal
	float3 dw; // ray dir of next path segment
	Refl_t refltype = REFR;
	// colour mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);
	bool isLight;
	intersect(r, scene_t, triangle_id, geomtype, pointHitInWorldSpace, isLight);

	// SPHERES
	// intersect all spheres in the scene
	float numspheres = sizeof(spheres) / sizeof(Sphere);
	for (int i = int(numspheres); i--;)  // for all spheres in scene
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(r)) && d < scene_t) { scene_t = d; sphere_id = i; geomtype = 1; }

	if (geomtype == 1) {
		Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
		x1 = r.origin + r.dir*scene_t;  // intersection point on object
		n = normalize(x - sphere.position);		// normal
		//nl = dot(n, r.dir) < 0 ? n : n * -1; // correctly oriented normal
		f = sphere.color;   // object colour
		refltype = sphere.refl;
		emit = sphere.emission;  // object emission
		//accucolor += (mask * emit);
	} else if (geomtype == 2) { // if Triangle:
		x1 = pointHitInWorldSpace;  // intersection point
		n = triangles[triangle_id].normal;  // normal 
		//nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal

		// colour, refltype and emit value are hardcoded and apply to all triangles
		// no per Triangle material support yet
		f = make_float3(1.0f);  // Triangle colour
		refltype = DIFF;
		if (isLight) emit = make_float3(1.0f);
		else emit = make_float3(0.0f);
		//accucolor += (mask * emit);
	}

	//we hit something
	if (geomtype == 2) {
		if (isLight) {
			buffer[index] += make_float4(1.0f);
		} else {
			//generate random point on light surface
			float lightArea = 1000;
			float lightWidth = 100;
			float3 lightPos = make_float3(0, 50, 50) + make_float3(lightWidth * (r1 * 2 - 1), 0, lightWidth * (r2 * 2 - 1));
			float3 lightNormal = make_float3(0, -1, 0);
			float3 lightColor = make_float3(10.0f);
			//check if we can see the light
			if (shadowRay(pointHitInWorldSpace, lightPos)) {
				float3 distance = lightPos - pointHitInWorldSpace;
				float3 l = normalize(distance);
				float cosineTerm = clamp(dot(n, -l), 0.0f, 1.0f);
				float projectedLightArea = clamp(dot(lightNormal, -l), 0.0f, 1.0f) * lightArea;
				float3 lightContribution = lightColor * f * cosineTerm * projectedLightArea / pow(length(distance), 2.0f) / M_PI;
				buffer[index] += make_float4(lightContribution, 1.0f);
			}
		}
	}

	// write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
//	buffer[index] += make_float4(r, 1.0f);
	float4 tempcol = buffer[index] / frameNumber;
	//tempcol = make_float4(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f), 1.0f);
	//tempcol /= 2.2f;
	surf2Dwrite(tempcol, surface, x * sizeof(float4), (height - y - 1));
}

// this hash function calculates a new random number generator seed for each frame, based on framenumber  
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

void render(Camera* cam, cudaSurfaceObject_t surface, float4* buffer, Triangle* dTriangles, LBVHNode* dNodes, const int numTriangles, unsigned int frameNumber) {
	unsigned int hashed = WangHash(frameNumber);

	dim3 block(32, 16);
	//dim3 grid(width / block.x, height / block.y, 1);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	
	//should do cudaCheck
	cudaCheck(cudaMemcpyToSymbol(triangles, &dTriangles, sizeof(float4*)));
	cudaCheck(cudaMemcpyToSymbol(nodes, &dNodes, sizeof(LBVHNode*)));

	//pathTrace<<<grid, block>>>(cam, surface, buffer, numTriangles, frameNumber, hashed);
	primaryRays<<<grid, block>>>(cam, surface, buffer, frameNumber, hashed);
}