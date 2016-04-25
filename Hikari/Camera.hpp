#ifndef CAMERA_H
#define CAMERA_H

#include <GL/glew.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "Ray.h"
#include <iostream>

#define M_PI 3.1415926535897932384626422832795028841971f

enum Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

class Camera {
public:
	int2 res;
	float2 fov;
	float3 eye, front, up;
	float3 hAxis, vAxis;
	float aperture, focalDistance;
	float yaw, pitch;
	float speed, mouseSens;
	bool moved;

	Camera(float3 position, int2 resolution, float fieldOfView, float apertureRadius, float focal) :
		   eye(position), res(resolution), aperture(apertureRadius), focalDistance(focal) {
		front = make_float3(0, 0, -1);
		up = make_float3(0, 1, 0);
		moved = false;
		yaw = -90.0f;
		pitch = 0.0f;
		speed = 10.0f;
		mouseSens = 0.1f;

		fov.x = fieldOfView;
		fov.y = (180 / M_PI) * (atanf(tanf(fieldOfView * 0.5f * (M_PI / 180) * ((float)resolution.y / (float)resolution.x)) * 2.0f));
	}

	void rebuildCamera() {
		float3 u = normalize(cross(front, up));
		float3 v = normalize(cross(u, front));

		hAxis = u * tanf(fov.x * 0.5f * (M_PI / 180));
		vAxis = v * tanf(-fov.y * 0.5f * (M_PI / 180));
	}

	void wasdMovement(Movement dir, float deltaTime) {
		moved = true;
		float velocity = speed * deltaTime;
		switch (dir) {
		case FORWARD:
			eye += front * velocity;
			break;
		case BACKWARD:
			eye -= front * velocity;
			break;
		case LEFT:
			eye -= normalize(cross(front, up)) * velocity;
			break;
		case RIGHT:
			eye += normalize(cross(front, up)) * velocity;
			break;
		case UP:
			eye.y += 2 * (float)velocity;
			break;
		case DOWN:
			eye.y -= 2 * (float)velocity;
			break;
		}
	}

	void mouseMovement(float xoffset, float yoffset, float deltaTime) {
		moved = true;
		yaw += (GLfloat)(mouseSens * deltaTime * xoffset);
		pitch += (GLfloat)(mouseSens * deltaTime * yoffset);

		if (pitch > 89.0f) pitch = 89.0f;
		if (pitch < -89.0f)	pitch = -89.0f;

		front.x = cosf(yaw) * cosf(pitch);
		front.y = sinf(pitch);
		front.z = sinf(yaw) * cosf(pitch);
		front = normalize(front);
	}

	__host__ __device__ Ray getRay(int x, int y, float r1, float r2) {
		float sx = ((r1 * 2 - 1) + x) / (res.x - 1);
		float sy = ((r2 * 2 - 1) + y) / (res.y - 1);

		float3 pointOnPlaneOneUnitAwayFromEye = eye + front + (hAxis * (sx * 2 - 1) + (vAxis * (sy * 2 - 1)));
		float3 pointOnImagePlane = eye + ((pointOnPlaneOneUnitAwayFromEye - eye) * focalDistance);

		//randomly pick a point on the circular aperture
		float angle = M_PI * 2 * r1;
		float distance = aperture * sqrtf(r2);
		float apertureX = cos(angle) * distance;
		float apertureY = sin(angle) * distance;

		float3 aperturePoint = eye + (hAxis * apertureX) + (vAxis * apertureY);

		float3 rayInWorldSpace = normalize(pointOnImagePlane - aperturePoint);
		return Ray(aperturePoint, rayInWorldSpace);
	}
};

#endif