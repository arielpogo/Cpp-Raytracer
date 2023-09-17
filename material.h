#ifndef MATERIAL_H
#define MATERIAL_H

#include "raytracing.h"

class hit_record; //prevent circular reference once hittable references material

class material{
public:
	virtual ~material() = default;

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
public:
	lambertian(const color& a) : albedo(a){}

	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override{
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
	metal(const color& a) : albedo(a){}

	bool scatter(const ray& r_in, const hit_record&rec, color& attenuation, ray& scattered) const override{
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
	solid(const color& a) : albedo(a){}

	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override{
		attenuation = albedo;
		return true;
	}

private:
	color albedo;
};

#endif
