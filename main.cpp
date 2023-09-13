#include "raytracing.h"

#include "camera.h"
#include "hittable_list.h"
#include "sphere.h"

int main(void){
	
	//World
	hittable_list world;
	world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
	world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));
	world.add(make_shared<sphere>(point3(0.25,0,-0.5), 0.1));
	
	camera cam;
	cam.image_width = 640;
	cam.aspect_ratio = 16.0/9.0;
	cam.samples_per_pixel = 16;
	cam.max_bounces = 10;
	cam.render(world);	

	return 0;
}
