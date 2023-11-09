#ifndef HITTABLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include "hittable.cuh"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {
public:
	//shared_ptr safely deletes itself once it goes out of scope
	std::vector<shared_ptr<hittable>> objects;

	__device__ __host__ hittable_list(){}
	__device__ __host__ hittable_list(shared_ptr<hittable> object) {add(object);}

	__device__ __host__ void clear() {objects.clear();}

	__device__ __host__ void add(shared_ptr<hittable> object) {
		objects.push_back(object);
	}

	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		hit_record temp_rec;
		bool hit_anything = false;
		double closest_yet = ray_t.max;

		for(const auto& object : objects){
			if(object->hit(r, interval(ray_t.min, closest_yet), temp_rec)){
				hit_anything = true;
				closest_yet = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}
};

#endif
