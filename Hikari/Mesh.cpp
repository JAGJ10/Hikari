#include "Mesh.h"
#include <string>
#include <fstream>
#include <iostream>
#include <stdint.h>

Mesh::Mesh(std::string file) : root() {
	if (!fileExists(file + ".cobj")) {
		std::ofstream outfile(file + ".cobj", std::ifstream::binary);
		write(file + ".obj", outfile);
		outfile.close();
		shapes.clear();
		materials.clear();
	}

	std::ifstream infile(file + ".cobj", std::ifstream::binary);
	read(infile);
	infile.close();

	float3 offset = make_float3(0, -2, 50);
	float3 scale = make_float3(-20, 20, 20);
	for (size_t i = 0; i < shapes.size(); i++) {
		for (int f = 0; f < shapes[i].mesh.indices.size(); f += 3) {
			int i1 = shapes[i].mesh.indices[f + 0];
			float3 v0 = make_float3(shapes[i].mesh.positions[i1 * 3], shapes[i].mesh.positions[i1 * 3 + 1], shapes[i].mesh.positions[i1 * 3 + 2]) * scale + offset;
			int i2 = shapes[i].mesh.indices[f + 1];
			float3 v1 = make_float3(shapes[i].mesh.positions[i2 * 3], shapes[i].mesh.positions[i2 * 3 + 1], shapes[i].mesh.positions[i2 * 3 + 2]) * scale + offset;
			int i3 = shapes[i].mesh.indices[f + 2];
			float3 v2 = make_float3(shapes[i].mesh.positions[i3 * 3], shapes[i].mesh.positions[i3 * 3 + 1], shapes[i].mesh.positions[i3 * 3 + 2]) * scale + offset;

			triBox triAABB((int)aabbs.size(), fminf(fminf(v0, v1), v2), fmaxf(fmaxf(v0, v1), v2));
			aabbs.push_back(triAABB);

			Triangle tri(make_float4(v0, 0), make_float4(v1, 0), make_float4(v2, 0));

			//edge vectors
			float3 vc1 = v1 - v0;
			float3 vc2 = v2 - v1;
			float3 vc3 = v0 - v2;

			// normal, cross product of edge vectors vc1 and vc2
			tri.normal = cross(vc1, vc2);
			tri.normal = normalize(tri.normal);
			triangles.push_back(tri);

			root.minBounds = fminf(root.minBounds, fminf(fminf(v0, v1), v2));
			root.maxBounds = fmaxf(root.maxBounds, fmaxf(fmaxf(v0, v1), v2));
		}
	}

	//Cleanup
	shapes.clear();
	materials.clear();
}

Mesh::~Mesh() {}

bool Mesh::fileExists(std::string fileName) {
	std::ifstream infile(fileName);
	return infile.good();
}

void Mesh::write(std::string fileName, std::ostream& stream) {
	assert(sizeof(float) == sizeof(uint32_t));
	const auto sz = sizeof(uint32_t);

	std::string err;
	bool ret = tinyobj::LoadObj(shapes, materials, err, fileName.c_str(), "objs/");
	std::cout << "Num Shapes: " << shapes.size() << std::endl;

	if (!err.empty()) { // `err` may contain warning message.
		std::cerr << err << std::endl;
	}

	if (!ret) exit(1);

	const uint32_t nMeshes = static_cast<uint32_t>(shapes.size());
	const uint32_t nMatProperties = 3;

	stream.write((const char*)&nMeshes, sz);        // nMeshes
	stream.write((const char*)&nMatProperties, sz); // nMatProperties

	for (size_t i = 0; i < nMeshes; ++i) {
		const uint32_t nVertices = (uint32_t)shapes[i].mesh.positions.size();
		const uint32_t nNormals = (uint32_t)shapes[i].mesh.normals.size();
		const uint32_t nTexcoords = (uint32_t)shapes[i].mesh.texcoords.size();
		const uint32_t nIndices = (uint32_t)shapes[i].mesh.indices.size();

		// Write nVertices, nNormals,, nTexcoords, nIndices
		// Write #nVertices positions
		// Write #nVertices normals
		// Write #nVertices texcoord
		// Write #nIndices  indices
		// Write #nMatProperties material properties
		stream.write((const char*)&nVertices, sz);
		stream.write((const char*)&nNormals, sz);
		stream.write((const char*)&nTexcoords, sz);
		stream.write((const char*)&nIndices, sz);

		stream.write((const char*)&shapes[i].mesh.positions[0], nVertices * sz);
		stream.write((const char*)&shapes[i].mesh.normals[0], nNormals * sz);
		stream.write((const char*)&shapes[i].mesh.texcoords[0], nTexcoords * sz);
		stream.write((const char*)&shapes[i].mesh.indices[0], nIndices * sz);
		//stream.write((const char*)&materials[i].ambient[0], 3 * sz);
		stream.write((const char*)&materials[i].diffuse[0], 3 * sz);
		//stream.write((const char*)&materials[i].specular[0], 3 * sz);
	}
}

void Mesh::read(std::istream& stream) {
	assert(sizeof(float) == sizeof(uint32_t));
	const auto sz = sizeof(uint32_t);

	uint32_t nMeshes = 0;
	uint32_t nMatProperties = 0;
	stream.read((char*)&nMeshes, sz);
	stream.read((char*)&nMatProperties, sz);
	shapes.resize(nMeshes);
	for (size_t i = 0; i < nMeshes; ++i) {
		uint32_t nVertices = 0, nNormals = 0, nTexcoords = 0, nIndices = 0;
		stream.read((char*)&nVertices, sz);
		stream.read((char*)&nNormals, sz);
		stream.read((char*)&nTexcoords, sz);
		stream.read((char*)&nIndices, sz);

		shapes[i].mesh.positions.resize(nVertices);
		shapes[i].mesh.normals.resize(nNormals);
		shapes[i].mesh.texcoords.resize(nTexcoords);
		shapes[i].mesh.indices.resize(nIndices);

		stream.read((char*)&shapes[i].mesh.positions[0], nVertices * sz);
		stream.read((char*)&shapes[i].mesh.normals[0], nNormals * sz);
		stream.read((char*)&shapes[i].mesh.texcoords[0], nTexcoords * sz);
		stream.read((char*)&shapes[i].mesh.indices[0], nIndices * sz);
		//stream.read((char*)&materials[i].ambient[0], 3 * sz);
		stream.read((char*)&materials[i].diffuse[0], 3 * sz);
		//stream.read((char*)&materials[i].specular[0], 3 * sz);
	}
}