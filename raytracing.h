#ifndef RAYTRACING_H
#define RAYTRACING_H

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "timing.cuh"

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

__constant__ double infinity = std::numeric_limits<double>::infinity();
#define PI 3.1415926535897932385

bool debug = false;

__host__ __device__ inline double degrees_to_radians(double degrees){
	return degrees * PI / 180.0;
}

__host__ __device__ inline double random_double(){
	return rand() / (RAND_MAX + 1.0); //[0,1)
}

__host__ __device__ inline double random_double(double min, double max){
	return min + (max-min)*random_double();
}

#include "ray.cuh"
#include "vec3.cuh"
#include "interval.cuh"

#endif
