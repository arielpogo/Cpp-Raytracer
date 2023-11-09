#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "raytracing.h"

class hit_record; //prevent circular reference once hittable references material

class material{
public:
	__device__ virtual ~material() = default;

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
public:
	__device__ lambertian(const color& a) : albedo(a){}

	__device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override{
		vec3 scatter_direction = rec.normal + random_unit_vector(); //this is what makes this lambertian

		if(scatter_direction.near_zero()) scatter_direction = rec.normal; //if the random unit vector nearly zeroes the reflection

		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo;
		return true;
	}
private:
	color albedo;
};

class metal : public material {
public:
	__device__ metal(const color& a) : albedo(a){}

	__device__ bool scatter(const ray& r_in, const hit_record&rec, color& attenuation, ray& scattered) const override{
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected);
		attenuation = albedo;
		return true;
	}
private:
	color albedo;	
};

//just for debug or whatever, this doesn't scatter light at all
class solid : public material {
public:
	__device__ solid(const color& a) : albedo(a){}

	__device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override{
		attenuation = albedo;
		return true;
	}

private:
	color albedo;
};

#endif
