#ifndef MESH_H
#define MESH_H

#include <vector>
#include "Triangle.hpp"
#include "tiny_obj_loader.h"

class Mesh {
public:
	std::vector<Triangle> triangles;
	std::vector<triBox> aabbs;
	AABB root;

	Mesh(std::string filePath, int start, float3 scale, float3 offset);
	~Mesh();

private:
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	bool fileExists(std::string fileName);
	void write(std::string fileName, std::ostream& stream);
	void read(std::istream& stream);
};

#endif