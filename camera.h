#ifndef CAMERA_H
#define CAMERA_H

#include "raytracing.h"

#include "color.h"
#include "hittable.h"
#include "material.h"

#include <iostream>


class camera{
public:
	//image
	int image_height = 720;
	double aspect_ratio = 16.0 / 9.0;
	double hfov = 90; //horizontal fov in degrees
	
	int samples_per_pixel = 10;
	int max_bounces = 10; //per ray
	
	point3 lookfrom = point3(0,0,0);
	point3 lookat = point3(0,0,0);
	vec3 relative_up = vec3(0,1,0);
	
	void render(const hittable& world){
		initialize();

		//P3 image format
		std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

		for(int j = 0; j < image_height; j++){
			std::clog << "\rProgress: " << (j*100)/image_height << "% " << std::flush;
			
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
	int image_width;
	point3 camera_center;
	point3 pixel00_loc;
	vec3 pixel_delta_u;
	vec3 pixel_delta_v;
	vec3 u, v, w; //camera vectors
	
	void initialize(){
		//calculate width based on aspect ratio and height, for user convenience
		image_width = static_cast<int>(aspect_ratio * image_height);
		image_width = (image_width < 1) ? 1 : image_width; //no 0px width image
		
		//viewport, camera
		camera_center = lookfrom;
		double focal_length = (lookfrom-lookat).length();

		//split the right triangle that is the view width into two, thus the hfov in two
		//tan(theta) = o/a, o is half of the width
		//so multiplying by a (focal length) gives the viewport width
		double viewport_width = 2*tan(degrees_to_radians(hfov)/2) * focal_length;
		double viewport_height = viewport_width/((double)image_width/(double)image_height);//recalculated because of int rounddown with the width, we want to be closer to what is calculated and not the perfect ratio
	
		//calculate vectors across the horz and down the vert frame
		w = unit_vector(lookfrom-lookat);
		u = unit_vector(cross(relative_up, w));
		v = cross(w,u);
	
		//keep track of the viewport's edges
		//-v because we render from topleft down, while
		//world is down to up
		vec3 viewport_u = viewport_width * u;
		vec3 viewport_v = viewport_height * -v;

		//calculate the distance between pixels in wordspace
		pixel_delta_u = viewport_u / image_width;
		pixel_delta_v = viewport_v / image_height;
		
		//calculate the location of the topleft pixel
		point3 viewport_upper_left = camera_center - (focal_length * w) - viewport_u/2 - viewport_v/2;
		pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

		if(debug){
			std::clog << "image w: " << image_width << '\n'
				  << "image h:" << image_height << '\n'
			          << "viewport w: " << viewport_width << '\n'
				  << "viewport h: " << viewport_height << '\n'
				  << "focal length: " << focal_length << '\n'
				  << std::endl;
		}
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
			ray scattered;
			color attenuation;
			
			if(rec.mat->scatter(r,rec,attenuation,scattered)) return attenuation * ray_color(scattered, depth-1, world);
			else return color(0,0,0);
		}

		//vec3 unit_direction = unit_vector(r.direction());
		//auto a = 0.5*(unit_direction.y() + 1.0);
		//return (1.0-a) * color (1,1,1) + a*color(0.5,0.5,0.9);
		return color(1.0,1.0,1.0);
	}
};

#endif
