#include "BVH.h"

BVH::BVH(const std::vector<triBox>& aabbs, const std::vector<Triangle>& triangles, AABB sceneBounds) {
	int totalNodes = 0;
	root = Recurse(aabbs, triangles, orderedTris, &totalNodes);
	root->min = sceneBounds.minBounds;
	root->max = sceneBounds.maxBounds;

	uint32_t offset = 0;
	nodes.resize(totalNodes);
	flattenBVH(root, nodes, &offset);

	cudaCheck(cudaMalloc((void**)&dNodes, nodes.size() * sizeof(LBVHNode)));
	cudaCheck(cudaMemcpy(dNodes, &nodes[0], nodes.size() * sizeof(LBVHNode), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&dTriangles, orderedTris.size() * sizeof(Triangle)));
	cudaCheck(cudaMemcpy(dTriangles, &orderedTris[0], orderedTris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));

	//Cleanup
	orderedTris.clear();
	deleteBVH(root);
}

BVH::~BVH() {
	cudaCheck(cudaFree(dNodes));
	cudaCheck(cudaFree(dTriangles));
}

void BVH::deleteBVH(BVHNode* n) {
	if (!n->IsLeaf()) {
		deleteBVH(dynamic_cast<BVHInner*>(n)->a);
		deleteBVH(dynamic_cast<BVHInner*>(n)->b);
	}

	delete n;
}

float BVH::calcSurfaceArea(const float3& extent) const {
	return extent.x*extent.y + extent.y*extent.z + extent.x*extent.z;
}

float BVH::getCenter(int axis, const triBox& v) const {
	float3 centroid = v.bb.getCentroid();
	if (axis == 0) return centroid.x;
	else if (axis == 1) return centroid.y;
	else return centroid.z;
}

void BVH::findBestSplit(const std::vector<triBox>& aabbs, int axis, int start, int end, float binSize, float& bestCost, float& bestSplit, int& bestAxis) {
	//for each bin (equally spaced bins of size "binSize"):
	for (float split = start + binSize; split < end - binSize; split += binSize) {
		// Create left and right bounding box
		float3 lMin = make_float3(FLT_MAX);
		float3 lMax = make_float3(-FLT_MAX);

		float3 rMin = make_float3(FLT_MAX);
		float3 rMax = make_float3(-FLT_MAX);

		int countLeft = 0, countRight = 0;

		//sort triangles into left and right boxes
		for (auto const& v : aabbs) {
			float value = getCenter(axis, v);

			if (value < split) {
				lMin = fminf(lMin, v.bb.minBounds);
				lMax = fmaxf(lMax, v.bb.maxBounds);
				countLeft++;
			} else {
				rMin = fminf(rMin, v.bb.minBounds);
				rMax = fmaxf(rMax, v.bb.maxBounds);
				countRight++;
			}
		}

		//check for bad splits
		if (countLeft <= 1 || countRight <= 1) continue;

		//calculate surface area of left and right boxes
		float3 lExtent = lMax - lMin;
		float3 rExtent = rMax - rMin;
		float splitCost = calcSurfaceArea(lExtent)*countLeft + calcSurfaceArea(rExtent)*countRight;

		//cheapest split
		if (splitCost < bestCost) {
			bestCost = splitCost;
			bestSplit = split;
			bestAxis = axis;
		}
	}
}

BVHLeaf* BVH::createLeaf(const std::vector<triBox>& aabbs, const std::vector<Triangle>& triangles, std::vector<Triangle>&orderedTris) {
	BVHLeaf* leaf = new BVHLeaf;
	leaf->triOffset = (uint32_t)orderedTris.size();
	leaf->triCount = (uint32_t)aabbs.size();
	for (auto const& v : aabbs) {
		orderedTris.push_back(triangles[v.index]);
	}
	return leaf;
}

BVHNode* BVH::Recurse(const std::vector<triBox>& aabbs, const std::vector<Triangle>& triangles, std::vector<Triangle>& orderedTris, int* totalNodes) {
	(*totalNodes)++;
	//if aabbs has less then 4 triangles, create a leaf node with the Triangle ids
	if (aabbs.size() < 4) {
		return createLeaf(aabbs, triangles, orderedTris);
	}

	float3 min = make_float3(FLT_MAX);
	float3 max = make_float3(-FLT_MAX);

	//expand bounding box
	for (auto const& v : aabbs) {
		min = fminf(min, v.bb.minBounds);
		max = fmaxf(max, v.bb.maxBounds);
	}

	//SAH cost = (number of triangles) * surfaceArea
	float3 extent = max - min;
	float bestCost = aabbs.size() * calcSurfaceArea(extent);
	float bestSplit = FLT_MAX;
	int bestAxis = -1;

	for (int axis = 0; axis < 3; axis++) {
		//divide triangles based on the current axis and split values from "start" to "end", one "binSize" at a time.
		float start, end, binSize;

		if (axis == 0) {
			start = min.x;
			end = max.x;
		} else if (axis == 1) {
			start = min.y;
			end = max.y;
		} else {
			start = min.z;
			end = max.z;
		}

		//box side along this axis too short, move to next axis
		if (fabsf(end - start) < 1e-4) continue;

		//Paper says 16 is enough bins, but should double check
		binSize = (end - start) / 16.f;
		//findBestSplit(aabbs, axis, start, end, binSize, bestCost, bestSplit, bestAxis);
		//for each bin (equally spaced bins of size "binSize"):
		for (float split = start + binSize; split < end - binSize; split += binSize) {
			// Create left and right bounding box
			float3 lMin = make_float3(FLT_MAX);
			float3 lMax = make_float3(-FLT_MAX);

			float3 rMin = make_float3(FLT_MAX);
			float3 rMax = make_float3(-FLT_MAX);

			int countLeft = 0, countRight = 0;

			//sort triangles into left and right boxes
			for (auto const& v : aabbs) {
				float value = getCenter(axis, v);

				if (value < split) {
					lMin = fminf(lMin, v.bb.minBounds);
					lMax = fmaxf(lMax, v.bb.maxBounds);
					countLeft++;
				} else {
					rMin = fminf(rMin, v.bb.minBounds);
					rMax = fmaxf(rMax, v.bb.maxBounds);
					countRight++;
				}
			}

			//check for bad splits
			if (countLeft <= 1 || countRight <= 1) continue;

			//calculate surface area of left and right boxes
			float3 lExtent = lMax - lMin;
			float3 rExtent = rMax - rMin;
			float splitCost = calcSurfaceArea(lExtent)*countLeft + calcSurfaceArea(rExtent)*countRight;

			//cheapest split
			if (splitCost < bestCost) {
				bestCost = splitCost;
				bestSplit = split;
				bestAxis = axis;
			}
		}
	}

	//if splitting isn't worth it, make a leaf
	if (bestAxis == -1) return createLeaf(aabbs, triangles, orderedTris);

	//split and recurse
	std::vector<triBox> left;
	std::vector<triBox> right;
	float3 lMin = make_float3(FLT_MAX);
	float3 lMax = make_float3(-FLT_MAX);
	float3 rMin = make_float3(FLT_MAX);
	float3 rMax = make_float3(-FLT_MAX);

	//place triangles in the left or right child nodes
	for (auto const& v : aabbs) {
		float value = getCenter(bestAxis, v);

		if (value < bestSplit) {
			left.push_back(v);
			lMin = fminf(lMin, v.bb.minBounds);
			lMax = fmaxf(lMax, v.bb.maxBounds);
		} else {
			right.push_back(v);
			rMin = fminf(rMin, v.bb.minBounds);
			rMax = fmaxf(rMax, v.bb.maxBounds);
		}
	}

	BVHInner* inner = new BVHInner;

	//recursively build the left child
	inner->a = Recurse(left, triangles, orderedTris, totalNodes);
	inner->a->min = lMin;
	inner->a->max = lMax;

	//recursively build the right child
	inner->b = Recurse(right, triangles, orderedTris, totalNodes);
	inner->b->min = rMin;
	inner->b->max = rMax;

	return inner;
}

uint32_t BVH::flattenBVH(BVHNode* n, std::vector<LBVHNode>& nodes, uint32_t* offset) {
	LBVHNode* lNode = &nodes[*offset];
	lNode->bb.minBounds = n->min;
	lNode->bb.maxBounds = n->max;
	uint32_t mOffset = (*offset)++;
	if (n->IsLeaf()) {
		BVHLeaf* p = dynamic_cast<BVHLeaf*>(n);
		lNode->triOffset = p->triOffset;
		lNode->triCount = p->triCount;
	} else {
		BVHInner* p = dynamic_cast<BVHInner*>(n);
		lNode->triCount = 0;
		flattenBVH(p->a, nodes, offset);
		lNode->rightChild = flattenBVH(p->b, nodes, offset);
	}

	return mOffset;
}