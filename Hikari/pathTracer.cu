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
#include "BVH.h"

#define M_PI  3.1415926535897932384626422832795028841971f

__device__ Triangle* triangles;
__device__ LBVHNode* nodes;

__device__ bool intersect(const Ray& r, HitInfo& hit) {
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
					if (t < hit.hitDist && t > 0.001f) {
						hit.hitDist = t;
						hit.tri = index + i;
						hit.hitPoint = r.origin + r.dir * t;
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
	HitInfo hit;
	if (!intersect(shadowRay, hit)) {
		return false;
	} else if (triangles[hit.tri].emit.x > 0.0f || triangles[hit.tri].emit.y > 0.0f || triangles[hit.tri].emit.z > 0.0f) {
		return true;
	}
	
	return false;
}

__global__ void secondaryRays(Ray* rays, int* activeRays, Camera* cam, float4* accumBuffer, float3* mask, unsigned int hash, int i, int numActiveRays) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= numActiveRays || activeRays[index] == -1) return;

	curandState randState;
	curand_init(hash + index, 0, 0, &randState);
	HitInfo hit = HitInfo();
	float3 c, emit, n;

	if (i == 0) mask[index] = make_float3(1.0f);

	if (intersect(rays[index], hit)) {
		//Make sure normal is oriented
		n = dot(triangles[hit.tri].normal, rays[index].dir) < 0 ? triangles[hit.tri].normal : triangles[hit.tri].normal * -1;
		c = triangles[hit.tri].diffuse;
		emit = triangles[hit.tri].emit;
		if (i > 0) {
			accumBuffer[index] += make_float4(mask[index] * emit, 1.0f);
		}
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
		rays[index].origin = hit.hitPoint + n * 0.01f;
	} else {
		activeRays[index] = -1;
	}
}

__global__ void primaryRays(Ray* rays, int* activeRays, Camera* cam, float4* buffer, unsigned int hash) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= (cam->res.x * cam->res.y)) return;

	activeRays[index] = index;

	curandState randState;
	curand_init(hash + index, 0, 0, &randState);

	float r1 = curand_uniform(&randState);
	float r2 = curand_uniform(&randState);
	rays[index] = cam->getRay(x, y, r1, r2);

	//Intersect primary Ray with scene and then do light sampling
	HitInfo hit;
	float3 c, emit, n;

	if (intersect(rays[index], hit)) {
		//Make sure normal is oriented
		n = dot(triangles[hit.tri].normal, rays[index].dir) < 0 ? triangles[hit.tri].normal : triangles[hit.tri].normal * -1;
		c = triangles[hit.tri].diffuse;
		emit = triangles[hit.tri].emit;

		if (emit.x > 0.0f || emit.y > 0.0f || emit.z > 0.0f) {
			buffer[index] += make_float4(c, 1.0f);
			activeRays[index] = -1;
		} else {
			//generate random point on light surface
			float lightArea = 225.0f;  //1000; 
			float lightWidth = 15.f / 2.f; //50;
			float lightHeight = 15.f / 2.f; //50; 
			float3 lightPos = make_float3(15.f, 27.4f, -15.f) + make_float3(lightWidth * (r1 * 2 - 1), 0, lightHeight * (r2 * 2 - 1));
			//float3 lightPos = make_float3(0, 55, 0) + make_float3(lightWidth * (r1 * 2 - 1), 0, lightHeight * (r2 * 2 - 1));
			float3 lightNormal = make_float3(0, -1, 0);
			float3 lightColor = make_float3(2.0f);
			//check if we can see the light
			float3 distance = lightPos - hit.hitPoint;
			float3 l = normalize(distance);
			if (shadowRay(hit.hitPoint, lightPos, l)) {				
				float cosineTerm = clamp(dot(n, l), 0.0f, 1.0f);
				float projectedLightArea = clamp(dot(lightNormal, -l), 0.0f, 1.0f) * lightArea;
				float3 lightContribution = lightColor * c * cosineTerm * projectedLightArea / pow(length(distance), 2.0f) / M_PI;
				buffer[index] += make_float4(lightContribution, 1.0f);
			}
		}
	} else {
		activeRays[index] = -1;
	}
}

__global__ void writePixels(Camera* cam, cudaSurfaceObject_t surface, float4* buffer, float4* accumBuffer, unsigned int frameNumber) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam->res.x);
	if (index >= (cam->res.x * cam->res.y)) return;

	buffer[index] += accumBuffer[index];
	float4 tempcol = buffer[index] / frameNumber;
	surf2Dwrite(tempcol, surface, x * sizeof(float4), (cam->res.y - y - 1));
}

struct isRayActive {
	__host__ __device__  bool operator()(const int& i) {
		return i < 0;
	}
};

//calculates a new random number generator seed for each frame, based on framenumber  
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

void render(const Camera& hostCam, Camera* cam, cudaSurfaceObject_t surface, float4* buffer, Triangle* dTriangles, LBVHNode* dNodes, unsigned int frameNumber, bool quickRender) {
	dim3 block(32, 32);
	dim3 grid((hostCam.res.x + block.x - 1) / block.x, (hostCam.res.y + block.y - 1) / block.y);
	int numActiveRays = hostCam.res.x * hostCam.res.y;
	
	unsigned int frameHash = WangHash(frameNumber);
	cudaCheck(cudaMemcpyToSymbol(triangles, &dTriangles, sizeof(float4*)));
	cudaCheck(cudaMemcpyToSymbol(nodes, &dNodes, sizeof(LBVHNode*)));

	Ray* rays;
	int* activeRays;
	cudaCheck(cudaMalloc((void**)&rays, numActiveRays * sizeof(Ray)));
	cudaCheck(cudaMalloc((void**)&activeRays, numActiveRays * sizeof(int)));

	float4* accumBuffer;
	float3* mask;
	cudaCheck(cudaMalloc((void**)&accumBuffer, numActiveRays * sizeof(float4)));
	cudaCheck(cudaMemset(accumBuffer, 0, numActiveRays * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&mask, numActiveRays * sizeof(float3)));
	cudaCheck(cudaMemset(mask, 0, numActiveRays * sizeof(float3)));

	primaryRays<<<grid, block>>>(rays, activeRays, cam, buffer, frameHash);
	if (!quickRender) {
		for (int i = 0; i < 4; i++) {
			unsigned int bounceHash = WangHash(i);
			/*thrust::device_ptr<int> tRays(activeRays);
			thrust::device_ptr<int> tRaysLast(thrust::remove_if(tRays, tRays + numActiveRays, isRayActive()));
			numActiveRays = tRaysLast.get() - activeRays;
			int sqrtNumActiveRays = (int)ceil(sqrtf(numActiveRays));
			dim3 compactGrid((sqrtNumActiveRays + block.x - 1) / block.x, (sqrtNumActiveRays + block.y - 1) / block.y);*/
			secondaryRays<<<grid, block>>>(rays, activeRays, cam, accumBuffer, mask, frameHash * bounceHash, i, numActiveRays);
		}
	}
	writePixels<<<grid, block>>>(cam, surface, buffer, accumBuffer, frameNumber);
	cudaCheck(cudaFree(rays));
	cudaCheck(cudaFree(activeRays));
	cudaCheck(cudaFree(accumBuffer));
	cudaCheck(cudaFree(mask));
}