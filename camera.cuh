#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "raytracing.h"

#include "color.cuh"
#include "color_255.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "material.cuh"

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
	
	__host__ int render(const hittable_list& world, std::string filename, int block_size_parameter) {
		initialize();

		const int num_pixels = image_height * image_width;

		std::ofstream output_file;

		output_file.open(filename + ".ppm");
		if (!output_file.is_open()) {
			std::cerr << "Fatal error: " << filename << ".ppm" << " could not be opened/created." << std::endl;
			return 1;
		}

		color_255* h_result = (color_255*) malloc(num_pixels * sizeof(color_255));

		if (h_result == NULL) {
			std::cerr << "Fatal error: " << num_pixels * sizeof(color_255) << " bytes of memory could not be allocated in host memory (RAM)." << std::endl;

			free(h_result);
			output_file.close();
			return 1;
		}

		color_255* d_result = nullptr;
		cudaMalloc((void**)&d_result, num_pixels * sizeof(color_255));

		int num_blocks = (int) std::ceil(num_pixels / (double) block_size_parameter);
		//int shm_size = 1024 * 48; //48KB

		render_kernel << <num_blocks, block_size_parameter >> > (world, d_result, *this);

		cudaMemcpy(h_result, d_result, num_pixels * sizeof(color_255), cudaMemcpyDeviceToHost);

		cudaFree(d_result);

		//P3 image format
		output_file << "P3\n" << image_width << ' ' << image_height << "\n255\n";

		for (int j = 0; j < image_height; j++) {
			for (int i = 0; i < image_width; i++) {
				int pixel = j * image_width + i;
				output_file << h_result[pixel].x() << ' ' << h_result[pixel].y() << ' ' << h_result[pixel].z() << '\n';
			}
		}
				
		std::clog << "\rDone.                                 \n";

		output_file.close();
		free(h_result);
	}

	//get initialized below
	int image_width;
	point3 camera_center;
	point3 pixel00_loc;
	vec3 pixel_delta_u;
	vec3 pixel_delta_v;
	vec3 u, v, w; //camera vectors

private:
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
					  << "image h: " << image_height << '\n'
			          << "viewport w: " << viewport_width << '\n'
				      << "viewport h: " << viewport_height << '\n'
					  << "focal length: " << focal_length << '\n'
				      << std::endl;
		}
	}
};

__global__ void render_kernel(const hittable& world, color_255* d_result, camera& cam) {
	//extern __shared__ color_255 sh_result;
	int pixel = threadIdx.x + (blockDim.x * blockIdx.x);
	int i = pixel % cam.image_width;
	int j = pixel / cam.image_width;

	color pixel_color(0, 0, 0);
	for (int sample = 0; sample < cam.samples_per_pixel; sample++) {
		ray r = get_ray(i, j, cam);
		pixel_color += ray_color(r, cam.max_bounces, world);
	}
	d_result[pixel] = color_255(pixel_color, cam.samples_per_pixel);
}

__device__ color ray_color(const ray& r, int depth, const hittable& world) {
	if (depth <= 0) return color(0, 0, 0); //limit recursion depth with max bounces  

	hit_record rec;
	if (world.hit(r, interval(0.001, infinity), rec)) { //if anything in the world is hit (ignoring floating point imprecision)
		ray scattered;
		color attenuation;

		if (rec.mat->scatter(r, rec, attenuation, scattered)) return attenuation * ray_color(scattered, depth - 1, world);
		else return color(0, 0, 0);
	}

	//vec3 unit_direction = unit_vector(r.direction());
	//auto a = 0.5*(unit_direction.y() + 1.0);
	//return (1.0-a) * color (1,1,1) + a*color(0.5,0.5,0.9);
	return color(1, 1, 1);
}

__device__ ray get_ray(int i, int j, camera& cam) {
	//get a randomly sampled camera ray for the given pixel
	vec3 pixel_center = cam.pixel00_loc + (i * cam.pixel_delta_u) + (j * cam.pixel_delta_v);
	vec3 pixel_sample = pixel_center + pixel_sample_square(cam);

	return ray(cam.camera_center, pixel_sample - cam.camera_center);
}

__device__ vec3 pixel_sample_square(camera& cam) {
	auto x = -0.5 + random_double();
	auto y = -0.5 + random_double();
	return (x * cam.pixel_delta_u) + (y * cam.pixel_delta_v);
}

#endif
