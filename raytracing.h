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

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

bool debug = false;

inline double degrees_to_radians(double degrees){
	return degrees * pi / 180.0;
}

inline double random_double(){
	return rand() / (RAND_MAX + 1.0); //[0,1)
}

inline double random_double(double min, double max){
	return min + (max-min)*random_double();
}

#include "ray.cuh"
#include "vec3.cuh"
#include "interval.cuh"

#endif
