#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"
#include "raytracing.h"

class material; //prevent circular reference

class hit_record{
public:
	point3 p;
	vec3 normal;
	shared_ptr<material> mat;	
	double t;
	bool front_face;
	
	//NOTE: outward_normal must have unit length
	//sets normal vector
	__device__ void set_face_normal(const ray &r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;	
	}
};

class hittable{
public:
	virtual ~hittable() = default;
	
	//Returns whether the hittable object/surface/whatever was hit within the min/max range, and modifies the provided hit_record with the hit information
	__device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif
