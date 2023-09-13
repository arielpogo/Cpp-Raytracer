#ifndef MATERIAL_H
#define MATERIAL_H

#include "raytracing.h"

class hit_record; //prevent circular reference once hittable references material

class material{
public:
	virtual ~material() = default;

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

#endif
