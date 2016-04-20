#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

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

__device__ bool intersect(const Ray& r, int& triId, float& hitDist, float3& hitPoint) {
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
					if (t < hitDist && t > 0.001f) {
						hitDist = t;
						triId = index + i;
						hitPoint = r.origin + r.dir * t;
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

__device__ bool shadowRay(float3 hitPoint, float3 lightPos, float3 rayDir) {
	Ray shadowRay(hitPoint + 0.01f * rayDir, rayDir);
	int tri = -1;
	float hitDist = 1e20;
	float3 point;
	if (!intersect(shadowRay, tri, hitDist, point)) {
		return false;
	} else if (triangles[tri].emit.x > 0.0f || triangles[tri].emit.y > 0.0f || triangles[tri].emit.z > 0.0f) {
		return true;
	}
	
	return false;
}

__global__ void secondaryRays(Ray* rays, Camera* cam, cudaSurfaceObject_t surface, float4* accumBuffer, float3* mask, unsigned int frameNumber, unsigned int hashed, int i) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= (cam->res.x * cam->res.y) || !rays[index].active) return;

	curandState randState;
	curand_init(hashed + index, 0, 0, &randState);
	//mask[index] = make_float3(1.0f);
	float3 accucolor = make_float3(0.0f);

	float hitDist = 1e20;
	int tri = -1;
	float3 hitPoint, c, emit, n;

	if (intersect(rays[index], tri, hitDist, hitPoint)) {
		//Make sure normal is oriented
		n = dot(triangles[tri].normal, rays[index].dir) < 0 ? triangles[tri].normal : triangles[tri].normal * -1;
		c = triangles[tri].diffuse;
		emit = triangles[tri].emit;
		accucolor += i > 0 ? mask[index] * emit : make_float3(0.0f);
		mask[index] *= c;

		//ideal diffuse reflection
		float r1 = 2 * M_PI * curand_uniform(&randState);
		float r2 = curand_uniform(&randState);
		float r2s = sqrtf(r2);

		//compute orthonormal coordinate frame uvw with hitpoint as origin 
		float3 u = normalize(cross((fabs(n.x) > .1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), n));
		float3 v = cross(n, u);

		//compute cosine weighted random Ray dir on hemisphere 
		rays[index].dir = normalize(u*cosf(r1)*r2s + v*sinf(r1)*r2s + n*sqrtf(1 - r2));
		rays[index].invDir = 1.0f / rays[index].dir;
		//offset origin next path segment to prevent self intersection
		rays[index].origin = hitPoint + n * 0.01f;
	} else {
		rays[index].active = false;
	}

	accumBuffer[index] += make_float4(accucolor, 1.0f);
}

__global__ void primaryRays(Ray* rays, Camera* cam, cudaSurfaceObject_t surface, float4* buffer, unsigned int frameNumber, unsigned int hashed) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= (cam->res.x * cam->res.y)) return;

	curandState randState;
	curand_init(hashed + index, 0, 0, &randState);

	float r1 = curand_uniform(&randState);
	float r2 = curand_uniform(&randState);
	rays[index] = cam->getRay(x, y, true, r1, r2);

	//Intersect primary Ray with scene and then do light sampling
	float hitDist = 1e20;
	int tri = -1;
	float3 hitPoint, c, emit, n;

	if (intersect(rays[index], tri, hitDist, hitPoint)) {
		//Make sure normal is oriented
		n = dot(triangles[tri].normal, rays[index].dir) < 0 ? triangles[tri].normal : triangles[tri].normal * -1;
		c = triangles[tri].diffuse;
		emit = triangles[tri].emit;

		if (emit.x > 0.0f || emit.y > 0.0f || emit.z > 0.0f) {
			buffer[index] += make_float4(c, 1.0f);
			rays[index].active = false;
		} else {
			//generate random point on light surface
			float lightArea = 1000;// 34.125f;
			float lightWidth = 50;// 6.5f / 2.f;
			float lightHeight = 50;// 5.25f / 2.f;
			//float3 lightPos = make_float3(13.9f, 27.4f, -13.975f) + make_float3(lightWidth * (r1 * 2 - 1), 0, lightHeight * (r2 * 2 - 1));
			float3 lightPos = make_float3(0, 55, 0) + make_float3(lightWidth * (r1 * 2 - 1), 0, lightHeight * (r2 * 2 - 1));
			float3 lightNormal = make_float3(0, -1, 0);
			float3 lightColor = make_float3(20.0f);
			//check if we can see the light
			float3 distance = lightPos - hitPoint;
			float3 l = normalize(distance);
			if (shadowRay(hitPoint, lightPos, l)) {				
				float cosineTerm = clamp(dot(n, l), 0.0f, 1.0f);
				float projectedLightArea = clamp(dot(lightNormal, -l), 0.0f, 1.0f) * lightArea;
				float3 lightContribution = lightColor * c * cosineTerm * projectedLightArea / pow(length(distance), 2.0f) / M_PI;
				buffer[index] += make_float4(lightContribution, 1.0f);
			}
		}
	} else {
		rays[index].active = false;
	}
}

__global__ void writePixels(Camera* cam, cudaSurfaceObject_t surface, float4* buffer, float4* accumBuffer, unsigned int frameNumber) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= (cam->res.x * cam->res.y)) return;

	float4 tempcol = (buffer[index] + accumBuffer[index]) / frameNumber;
	surf2Dwrite(tempcol, surface, x * sizeof(float4), (height - y - 1));
}

struct isRayActive {
	__host__ __device__  bool operator()(const Ray& r) {
		return r.active == -1;
	}
};

// this hash function calculates a new random number generator seed for each frame, based on framenumber  
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

void render(Camera* cam, cudaSurfaceObject_t surface, float4* buffer, Triangle* dTriangles, LBVHNode* dNodes, unsigned int frameNumber, bool quickRender) {
	dim3 block(32, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	int numActiveRays = width * height;
	int launchLimit = (width * height) / 16;
	
	unsigned int hashed = WangHash(frameNumber);
	cudaCheck(cudaMemcpyToSymbol(triangles, &dTriangles, sizeof(float4*)));
	cudaCheck(cudaMemcpyToSymbol(nodes, &dNodes, sizeof(LBVHNode*)));

	Ray* rays;
	cudaCheck(cudaMalloc((void**)&rays, sizeof(Ray) * width * height));

	float4* accumBuffer;
	float3* mask;
	cudaCheck(cudaMalloc((void**)&accumBuffer, width * height * sizeof(float4)));
	cudaCheck(cudaMemset(accumBuffer, 0, width * height * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&mask, width * height * sizeof(float3)));
	cudaCheck(cudaMemset(mask, 1, width * height * sizeof(float3)));

	primaryRays<<<grid, block>>>(rays, cam, surface, buffer, frameNumber, hashed);
	if (!quickRender) {
		for (int i = 0; i < 3; i++) {
			thrust::device_ptr<Ray> tRays = thrust::device_pointer_cast(rays);
			thrust::device_ptr<Ray> tRaysLast = thrust::remove_if(tRays, tRays + numActiveRays, isRayActive());
			numActiveRays = thrust::distance(tRays, tRaysLast);
			dim3 compactGrid((sqrt(numActiveRays) + block.x - 1) / block.x, (sqrt(numActiveRays) + block.y - 1) / block.y);
			secondaryRays<<<compactGrid, block>>>(rays, cam, surface, accumBuffer, mask, frameNumber, hashed, i);
		}
	}
	writePixels<<<grid, block>>>(cam, surface, buffer, accumBuffer, frameNumber);
	cudaCheck(cudaFree(rays));
	cudaCheck(cudaFree(accumBuffer));
	cudaCheck(cudaFree(mask));
}