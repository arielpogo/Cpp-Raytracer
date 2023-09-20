#include "raytracing.h"

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

int main(int argc, char* argv[]){
	int height_parameter = 0;
	int ratio_parameter = 0;
	//Handle command line arguments
	if(argc > 1){ //if any arguments (beyond executable name)
		for(int i = 1; i < argc; i++){
			char* str = argv[i];

			if(str[0] == '-'){ //if this is an argument
				switch(str[1]){

					case 'i':
						std::clog << "-i: display this menu\n-d: enable debug\n-o <name>: specify output file name\n-h <int>: specifiy image height\n-r <int> specify preset ratio (4 = 4:3, 16 = 16:9, 10 = 16:10, 1 = 1:1)" << std::endl;
						return 0;
						break;

					case 'd': //enable debug
						debug = true;
						std::clog << "Debug enabled." << std::endl;
					break;

					case 'o': //set output file name
						i++; //go to the parameter

						if (i < argc) str = argv[i];
						else continue;

						if (str[0] != '-') filename.assign(str); //if there is a parameter, assign it to the file name
						else i--; //otherwise go back to this argument, so the next one is handled
					break;

					case 'h': //set image height
						i++; //go to the parameter

						if (i < argc) str = argv[i];
						else continue;

						if (str[0] != '-') height_parameter = atoi(str); //if there is a parameter, assign it to the file name
						else i--; //otherwise go back to this argument, so the next one is handled
					break;

					case 'r':
						i++; //go to the parameter

						if (i < argc) str = argv[i];
						else continue;

						if (str[0] != '-') ratio_parameter = atoi(str); //if there is a parameter, assign it to the file name
						else i--; //otherwise go back to this argument, so the next one is handled
					break;
				}
			}	
		}
	}

	output_file.open(filename + ".ppm");
	if (!output_file.is_open()) {
		std::cerr << "Error: " << filename << ".ppm" << " could not be opened/created." << std::endl;
		return 1;
	}

	hittable_list world;

	auto material_ground = make_shared<lambertian>(color(0.0,148.0/255.0,132.0/255.0));
	auto material_center = make_shared<lambertian>(color(0.8, 1, 0));
	auto material_right  = make_shared<metal>(color(1.0, 216.0/255.0, 0.0));

	world.add(make_shared<sphere>(point3( 0.0, -100.5, 0), 100.0, material_ground));
	world.add(make_shared<sphere>(point3( -1.0,    0.5, 0),   0.5, material_center));
	world.add(make_shared<sphere>(point3( 1.0,    0.0, 0),   0.6, material_right));
	
	camera cam;
	cam.image_height = (height_parameter > 0) ? height_parameter : cam.image_height;
	cam.hfov = 90;

	//pick aspect ratio
	if (ratio_parameter == 10) cam.aspect_ratio = 16.0 / 10.0;
	else if (ratio_parameter == 1) cam.aspect_ratio = 1.0;
	else if (ratio_parameter == 4) cam.aspect_ratio = 4.0 / 3.0;
	else cam.aspect_ratio = 16.0 / 9.0;
	
	cam.samples_per_pixel = 100;
	cam.max_bounces = 50;

    cam.lookfrom = point3(0,2,-5);
    cam.lookat   = point3(0,0,0);
	
	cam.render(world);	

	output_file.close();

	return 0;
}
