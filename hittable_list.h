#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {
public:
	//shared_ptr safely deletes itself once it goes out of scope
	std::vector<shared_ptr<hittable>> objects;

	hittable_list(){}
	hittable_list(shared_ptr<hittable> object) {add(object);}

	void clear() {objects.clear();}

	void add(shared_ptr<hittable> object) {
		objects.push_back(object);
	}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
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
