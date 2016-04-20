#include <Windows.h>
#define GLEW_DYNAMIC
#include <GL/glew.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <chrono>
#include "bvh.h"
#include "helper_math.h"
#include "Camera.hpp"
#include "FullscreenQuad.h"
#include "Shader.h"
#include "Mesh.h"

#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

void handleInput(GLFWwindow* window, Camera &cam);
void render(Camera* cam, cudaSurfaceObject_t surface, float4* buffer, Triangle* dTriangles, LBVHNode* dNodes, unsigned int frameNumber, bool quickRender);

static const int width = 1280;
static const int height = 720;
static const GLfloat lastX = (width / 2);
static const GLfloat lastY = (height / 2);
static float deltaTime = 0.0f;
static float lastFrame = 0.0f;
unsigned int frameNumber = 0;

void createFile(BVHNode* node, std::ofstream& myfile) {
	if (!node->IsLeaf()) {
		myfile << node->min.x << " " << node->min.y << " " << node->min.z << " " << node->max.x << " " << node->max.y << " " << node->max.z << std::endl;
		createFile(dynamic_cast<BVHInner*>(node)->a, myfile);
		createFile(dynamic_cast<BVHInner*>(node)->b, myfile);
	}

	/*for (auto const& n : nodes) {
		if (n.triCount == 0) {
			myfile << n.bb.minBounds.x << " " << n.bb.minBounds.y << " " << n.bb.minBounds.z << " " << n.bb.maxBounds.x << " " << n.bb.maxBounds.y << " " << n.bb.maxBounds.z << std::endl;
		}
	}*/
}

int main() {
	//Checks for memory leaks in debug mode
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Hikari", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	//Set callbacks for keyboard and mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glewExperimental = GL_TRUE;
	glewInit();
	glGetError();

	//Define the viewport dimensions
	glViewport(0, 0, width, height);

	//Initialize cuda->opengl context
	cudaCheck(cudaGLSetGLDevice(0));
	cudaGraphicsResource *resource;

	// create the texture that we use to visualize the ray-tracing result
	GLuint tex;
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	cudaCheck(cudaGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
	glBindTexture(GL_TEXTURE_2D, 0);

	Shader final = Shader("fsQuad.vert", "fsQuad.frag");
	FullscreenQuad fsQuad = FullscreenQuad();

	float4* buffer;
	cudaCheck(cudaMalloc((void**)&buffer, width * height * sizeof(float4)));
	cudaCheck(cudaMemset(buffer, 0, width * height * sizeof(float4)));

	//Mesh
	float3 offset = make_float3(0);
	float3 scale = make_float3(-15, 15, -15);
	Mesh cBox("objs/Avent", 0, scale, offset);
	offset = make_float3(0, 55, 0);
	scale = make_float3(100);
	Mesh light("objs/plane", (int)cBox.triangles.size(), scale, offset);
	cBox.triangles.insert(cBox.triangles.end(), light.triangles.begin(), light.triangles.end());
	cBox.aabbs.insert(cBox.aabbs.end(), light.aabbs.begin(), light.aabbs.end());
	std::cout << "Num triangles: " << cBox.triangles.size() << std::endl;
	cBox.root = AABB(fminf(cBox.root.minBounds, light.root.minBounds), fmaxf(cBox.root.maxBounds, light.root.maxBounds));
	BVH bvh(cBox.aabbs, cBox.triangles, cBox.root);

	//std::ofstream myfile;
	//myfile.open("Avent.txt");
	//createFile(root, myfile);
	//myfile.close();

	Camera cam(make_float3(14, 15, 80), make_int2(width, height), 45.0f, 0.04f, 100.0f);
	Camera* dCam;

	cudaCheck(cudaMalloc((void**)&dCam, sizeof(Camera)));
	cudaCheck(cudaMemcpy(dCam, &cam, sizeof(Camera), cudaMemcpyHostToDevice));

	cudaCheck(cudaGraphicsMapResources(1, &resource, 0));
	cudaArray* pixels;
	cudaCheck(cudaGraphicsSubResourceGetMappedArray(&pixels, resource, 0, 0));
	cudaResourceDesc viewCudaArrayResourceDesc;
	viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
	viewCudaArrayResourceDesc.res.array.array = pixels;
	cudaSurfaceObject_t viewCudaSurfaceObject;
	cudaCheck(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
	cudaCheck(cudaGraphicsUnmapResources(1, &resource, 0));

	while (!glfwWindowShouldClose(window)) {
		float currentFrame = float(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		//Check and call events
		glfwPollEvents();
		handleInput(window, cam);

		if (cam.moved) {
			frameNumber = 0;
			cudaCheck(cudaMemset(buffer, 0, width * height * sizeof(float4)));
		}

		cam.rebuildCamera();
		cudaCheck(cudaMemcpy(dCam, &cam, sizeof(Camera), cudaMemcpyHostToDevice));

		frameNumber++;
				
		if (frameNumber < 20000) {
			cudaCheck(cudaGraphicsMapResources(1, &resource, 0));
			std::chrono::time_point<std::chrono::system_clock> start, end;
			start = std::chrono::system_clock::now();
			render(dCam, viewCudaSurfaceObject, buffer, bvh.dTriangles, bvh.dNodes, frameNumber, cam.moved);
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed = end - start;
			std::cout << "Frame: " << frameNumber << " --- Elapsed time: " << elapsed.count() << "s\n";
			cudaCheck(cudaGraphicsUnmapResources(1, &resource, 0));
		}

		cam.moved = false;

		glUseProgram(final.program);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);

		glClear(GL_COLOR_BUFFER_BIT);
		
		final.setUniformi("tRender", 0);
		fsQuad.render();

		//std::cout << glGetError() << std::endl;

		//Swap the buffers
		glfwSwapBuffers(window);

		glfwSetCursorPos(window, lastX, lastY);
	}

	//Cleanup
	cudaCheck(cudaDeviceSynchronize());
	cudaCheck(cudaDestroySurfaceObject(viewCudaSurfaceObject));
	cudaCheck(cudaGraphicsUnregisterResource(resource));
	cudaCheck(cudaFree(buffer));
	cudaCheck(cudaFree(dCam));
	glDeleteTextures(1, &tex);

	glfwTerminate();

	return 0;
}

void handleInput(GLFWwindow* window, Camera &cam) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cam.wasdMovement(FORWARD, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cam.wasdMovement(BACKWARD, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cam.wasdMovement(RIGHT, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cam.wasdMovement(LEFT, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cam.wasdMovement(UP, deltaTime);

	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		cam.wasdMovement(DOWN, deltaTime);

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
		cam.mouseMovement((float(xpos) - lastX), (lastY - float(ypos)), deltaTime);
}