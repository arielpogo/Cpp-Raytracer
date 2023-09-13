#ifndef CAMERA_H
#define CAMERA_H

#include "raytracing.h"

#include "color.h"
#include "hittable.h"

#include <iostream>


class camera{
public:
	//image
	int image_width = 1280;
	double aspect_ratio = 16.0 / 9.0;
	int samples_per_pixel = 10;
	int max_bounces = 10; //per ray
	
	void render(const hittable& world){
		initialize();
		
		//P3 image format
		std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

		for(int j = 0; j < image_height; j++){
			std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
			
			for(int i = 0; i < image_width; i++){
				color pixel_color(0,0,0);
				for(int sample = 0; sample < samples_per_pixel; sample++){
					ray r = get_ray(i,j);
					pixel_color += ray_color(r, max_bounces, world);
				}
				write_color(std::cout, pixel_color, samples_per_pixel);
			}
		}

		std::clog << "\rDone.                                 \n";
	}

private:
	int image_height;
	point3 camera_center;
	point3 pixel00_loc;
	vec3 pixel_delta_u;
	vec3 pixel_delta_v;
	
	
	void initialize(){
		//calculate height based on the aspect ratio for simplicity
		image_height = static_cast<int>(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height; //no 0px height images
		
		//viewport, camera
		double focal_length = 1.0;
		double viewport_height = 2.0;
		double viewport_width = viewport_height * (static_cast<double>(image_width)/image_height); //recalculated because of the calculations done to image_height potentially deviating from  aspect_ratio
		camera_center = point3(0,0,0);
		
		//vectors to keep track of viewport edges. Top left of viewport is y = 0, but physically is a higher y, thus we use this to convert
		vec3 viewport_u = vec3(viewport_width, 0, 0);
		vec3 viewport_v = vec3(0, -viewport_height, 0);

		//calculate the distance between pixels in wordspace
		pixel_delta_u = viewport_u / image_width;
		pixel_delta_v = viewport_v / image_height;
		
		//calculate the location of the topleft pixel
		point3 viewport_upper_left = camera_center - vec3(0,0,focal_length) - viewport_u/2 - viewport_v/2;
		pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);	
	}

	ray get_ray(int i, int j){
		//get a randomly sampled camera ray for the given pixel
		vec3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
		vec3 pixel_sample = pixel_center + pixel_sample_square();

		return ray(camera_center, pixel_sample-camera_center);		
	}

	vec3 pixel_sample_square() const{
		auto x = -0.5 + random_double();
		auto y = -0.5 + random_double();
		return (x * pixel_delta_u) + (y * pixel_delta_v);
	}

	color ray_color(const ray& r, int depth, const hittable& world){
		if(depth <= 0) return color(0,0,0); //limit recursion depth with max bounces  

		hit_record rec;
		if(world.hit(r, interval(0.001,infinity), rec)){ //if anything in the world is hit (ignoring floating point imprecision)
			vec3 direction = random_on_hemisphere(rec.normal);
			return 0.5 * ray_color(ray(rec.p, direction), depth-1, world); //return the bounce of a bounce of a bounce....
		}

		vec3 unit_direction = unit_vector(r.direction());
		auto a = 0.5*(unit_direction.y() + 1.0);
		return (1.0-a) * color (1,0.5,0) + a*color(0.7,0.4,0);
	}
};

#endif
