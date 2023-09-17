#include "raytracing.h"

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

int main(int argc, char* argv[]){
	if(argc > 1){
		for(int i = 1; i < argc; i++){
			char* str = argv[i];
			if(i%2 == 1 && str[0] == '-'){
				switch(str[1]){
					case 'd':
						debug = true;
						std::clog << "Debug enabled." << std::endl;
					break;
					case 'o':
					break;
				}
			}	
		}
	}

	hittable_list world;

	auto material_ground = make_shared<lambertian>(color(0.05, 0.05, 0.05));
	auto material_center = make_shared<lambertian>(color(0.8, 1, 0));
	auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2));

	world.add(make_shared<sphere>(point3( 0.0, -100.5, 0), 100.0, material_ground));
	world.add(make_shared<sphere>(point3( -1.0,    0.0, 0),   0.5, material_center));
	world.add(make_shared<sphere>(point3( 1.0,    0.0, 0),   0.5, material_right));
	
	camera cam;
	cam.image_height = 720;
	cam.aspect_ratio = 16.0/9.0;
	cam.hfov = 40;
	
	cam.samples_per_pixel = 100;
	cam.max_bounces = 50;
	
	
    	cam.lookfrom = point3(0,2,-5);
        cam.lookat   = point3(0,0,0);
	
	cam.render(world);	

	return 0;
}
