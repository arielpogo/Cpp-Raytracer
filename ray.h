#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray{
public:
	//default constructor
	ray(){}
	
	//parameterized constructor
	ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

	//accessors
	point3 origin() const { return orig; }
	vec3 direction() const { return dir; }
	
	//a given point along the ray, t in the equation below
	point3 at(double t) const{
		return orig + t*dir;
	}

private:
	//A ray is represented as
	//A + tB
	point3 orig; //origin, A
	vec3 dir; //direction, B
};

#endif
