#include "raytracing.h"

#include "camera.cuh"
#include "color.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "sphere.cuh"

__global__ void init_kernel(hittable** d_list, int NUM_OBJECTS, hittable* d_world, material** d_materials, int NUM_MATERIALS, camera* d_cam, int height_parameter, int ratio_parameter) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	auto material_ground = new lambertian(color(0.5, 0.5, 0.5));
	auto material_center = new lambertian(color(0.8, 1, 0));
	auto material_right = new metal(color(1.0, 216.0 / 255.0, 0.0));

	d_list[0] = new sphere(point3(0.0, -100.5, 0), 100.0, material_ground);
	d_list[1] = new sphere(point3(-1.0, 0.5, 0), 0.5, material_center);
	d_list[2] = new sphere(point3(1.0, 0.0, 0), 0.6, material_right);

	d_world = new hittable_list(d_list, NUM_OBJECTS);

	d_cam->image_height = (height_parameter > 0) ? height_parameter : d_cam->image_height;
	d_cam->hfov = 90;

	//pick aspect ratio
	if (ratio_parameter == 10) d_cam->aspect_ratio = 16.0 / 10.0;
	else if (ratio_parameter == 1) d_cam->aspect_ratio = 1.0;
	else if (ratio_parameter == 4) d_cam->aspect_ratio = 4.0 / 3.0;
	else d_cam->aspect_ratio = 16.0 / 9.0;

	d_cam->samples_per_pixel = 100;
	d_cam->max_bounces = 50;

	d_cam->lookfrom = point3(0, 2, -5);
	d_cam->lookat = point3(0, 0, 0);
}

__global__ void free_kernel(hittable* d_world, material** d_materials, int NUM_MATERIALS, camera* d_cam) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	for (int i = 0; i < NUM_MATERIALS; i++) delete d_materials[i];
	delete d_materials;
	delete d_world;
	delete d_cam;
}

__host__ int render(hittable* d_world, material** d_materials, camera* d_cam, std::string filename, int block_size_parameter) {
	d_cam->initialize();

	const int num_pixels = d_cam->image_height * d_cam->image_width;

	std::ofstream output_file;

	output_file.open(filename + ".ppm");
	if (!output_file.is_open()) {
		std::cerr << "Fatal error: " << filename << ".ppm" << " could not be opened/created." << std::endl;
		return 1;
	}

	color_255 h_result[] = color_255[num_pixels];

	if (h_result == NULL) {
		std::cerr << "Fatal error: " << num_pixels * sizeof(color_255) << " bytes of memory could not be allocated in host memory (RAM)." << std::endl;

		free(h_result);
		output_file.close();
		return 1;
	}

	color_255* d_result = nullptr;
	cudaMalloc((void**)&d_result, num_pixels * sizeof(color_255));

	int num_blocks = (int)std::ceil(num_pixels / (double)block_size_parameter);
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

int main(int argc, char* argv[]){
	int height_parameter = 0;
	int ratio_parameter = 0;
	std::string filename = "image";
	int block_size_parameter = 32; //default
	const int NUM_OBJECTS = 3;
	const int NUM_MATERIALS = 3;

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

					case 'b': //block size
						i++; //go to parameter

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

	hittable** d_list = nullptr;
	cudaMalloc((void**)&d_list, NUM_OBJECTS * sizeof(hittable*));

	hittable* d_world;
	cudaMalloc((void**)&d_world, sizeof(hittable_list));

	material** d_materials = nullptr;
	cudaMalloc((void**)&d_materials, NUM_MATERIALS * sizeof(material*));

	camera* d_cam;
	cudaMalloc((void**)&d_cam, sizeof(camera));

	init_kernel<<<1, 1>>>(d_list, NUM_OBJECTS, d_world, d_materials, NUM_MATERIALS, d_cam, height_parameter, ratio_parameter);
	cudaFree(d_list);

	TIMING_START();

	int render_result = render(d_world, d_materials, d_cam, filename, block_size_parameter);

	TIMING_STOP();
	TIMING_PRINT();

	free_kernel <<<1, 1 >>> (d_world, d_materials, NUM_MATERIALS, d_cam);

	return render_result;
}
