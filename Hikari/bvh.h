#ifndef BVH_H
#define BVH_H

#include "Triangle.hpp"
#include <vector>
#include <stdint.h>

struct BVHNode {
	float3 min, max;
	virtual bool IsLeaf() = 0; //pure virtual
};

struct BVHInner : BVHNode {
	BVHNode* a;
	BVHNode* b;
	virtual bool IsLeaf() { return false; }
};

struct BVHLeaf : BVHNode {
	uint32_t triOffset;
	uint32_t triCount;
	virtual bool IsLeaf() { return true; }
};

struct LBVHNode {
	AABB bb;
	uint32_t triCount; //0 means inner
	union {
		uint32_t triOffset;
		uint32_t rightChild;
	};
};

class BVH {
public:
	Triangle* dTriangles;
	std::vector<Triangle> orderedTris;
	LBVHNode* dNodes;

	BVH(const std::vector<triBox>& aabbs, const std::vector<Triangle>& triangles, AABB sceneBounds);
	~BVH();

private:
	BVHNode* root;
	std::vector<LBVHNode> nodes;

	BVHNode* Recurse(const std::vector<triBox>& aabbs, const std::vector<Triangle>& triangles, std::vector<Triangle>& orderedTris, int* totalNodes);
	BVHLeaf* createLeaf(const std::vector<triBox>& aabbs, const std::vector<Triangle>& triangles, std::vector<Triangle>&orderedTris);
	void findBestSplit(const std::vector<triBox>& aabbs, int axis, int start, int end, float binSize, float& bestCost, float& bestSplit, int& bestAxis);
	float calcSurfaceArea(const float3& extent) const;
	float getCenter(int axis, const triBox& v) const;
	uint32_t flattenBVH(BVHNode* n, std::vector<LBVHNode>& nodes, uint32_t* offset);
	void deleteBVH(BVHNode* n);
};

#endif