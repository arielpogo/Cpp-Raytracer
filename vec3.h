#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

using std::sqrt;

class vec3{
public:
	double e[3];

	//default constructor
	vec3() : e{0,0,0} {

	}
	
	//parameterized constructor
	vec3(double e0, double e1, double e2) : e{e0, e1, e2} {
		
	}

	//accessors
	double x() const { return e[0]; }
	double y() const { return e[1]; }
	double z() const { return e[2]; }

	//simple operators, negation, access
	vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	double operator[](int i) const { return e[i]; }
	double& operator[](int i) { return e[i]; }
	
	//add other vector's stuff to this and return this as the answer
	vec3& operator+=(const vec3 &v){
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	//ditto, but with another double
	vec3& operator*=(double t){
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}
	
	//ditto, using fractions
	vec3& operator/=(double t){
		return *this *= 1/t;
	}

	double length() const{
		return sqrt(length_squared());
	}

	double length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
	
	static vec3 random(){
		return vec3(random_double(), random_double(), random_double());
	}

	static vec3 random(double min, double max){
		return vec3(random_double(min, max), random_double(min,max), random_double(min,max));
	}
};

//point3 is an alias of vec3
using point3 = vec3;

//allow vec3 to be printed, written to a file etc., especially for PPM files
inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

//add vec3s together
inline vec3 operator+(const vec3 &u, const vec3 &v){
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

//subtract vec3s
inline vec3 operator-(const vec3 &u, const vec3 &v){
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

//multiply vec3s
inline vec3 operator*(const vec3 &u, const vec3 &v){
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

//multiply a double with a vec3
inline vec3 operator*(double t, const vec3 &v){
	return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

//multiply a vec3 with a double
inline vec3 operator*(const vec3 &v, double t){
	return t * v;
}

//divide a vec3 by a double
inline vec3 operator/(vec3 v, double t) {
	return (1/t) * v;
}

//return the dot product of two vec3s
inline double dot(const vec3 &u, const vec3 &v){
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

//return the cross product of two vec3s
inline vec3 cross(const vec3 &u, const vec3 &v){
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		    u.e[2] * v.e[0] - u.e[0] * v.e[2],
		    u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

//return the unit vector of a vec3
inline vec3 unit_vector(vec3 v){
	return v / v.length();
}

//produces a random vec3 which is in a sphere of 1,1,1
inline vec3 random_in_unit_sphere(){
	while(true){
		//generating vectors in a sphere is difficult, the simplist way is to keep generating within a box until they are
		//a length of 1 (within a sphere if placed in the center, longer than 1 is outside a sphere
		//also we can't shrink the box within the sphere or else there would be some direction impossible to generate (circled square vs squared circle)
		vec3 p = vec3::random(-1,1);
		if(p.length_squared() < 1) return p;
	}
}

inline vec3 random_unit_vector(){
	return unit_vector(random_in_unit_sphere());
}

//a random vec3 within the same hemisphere as a given vec3, usually a surface normal
//one application is diffuse materials where we reflect off of the surface in a random
//direction in the same hemisphere as the normal (thus not into the sphere itself)
//we can tell by seeing if dot prod. is positive, if not then flip
inline vec3 random_on_hemisphere(const vec3& normal){
	vec3 on_unit_sphere = random_unit_vector();
	if(dot(on_unit_sphere, normal) > 0.0) return on_unit_sphere;
	else return -on_unit_sphere;
}

#endif
