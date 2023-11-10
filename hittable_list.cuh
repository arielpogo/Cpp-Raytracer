#ifndef HITTABLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include "hittable.cuh"

class hittable_list : public hittable {
public:
	hittable** objects;
	int n_objects;

	__device__ __host__ hittable_list(){}
	__device__ __host__ hittable_list(hittable** list, int n) : objects(list), n_objects(n) {};
	__device__ __host__ ~hittable_list() {
		for (int i = 0; i < n_objects; i++) delete objects[i];
	}

	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		hit_record temp_rec;
		bool hit_anything = false;
		double closest_yet = ray_t.max;

		for (int i = 0; i < n_objects; i++) {
			if(objects[i]->hit(r, interval(ray_t.min, closest_yet), temp_rec)) {
				hit_anything = true;
				closest_yet = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}
};

#endif
