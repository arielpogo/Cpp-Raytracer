#ifndef INTERVAL_CUH
#define INTERVAL_CUH

class interval{
public:
	double min, max;

	__host__ __device__ interval() : min(+infinity), max(-infinity) {} //empty on default
	__host__ __device__ interval(double _min, double _max) : min(_min), max(_max) {}

	//whether x is in range
	__host__ __device__ bool contains(double x) const {
		return min <= x && x <= max;
	}
	
	//whether x is in range, exclusive
	__host__ __device__ bool surrounds(double x) const {
		return min < x && x < max;
	}

	__host__ __device__ double clamp(double x) const{
		if(x < min) return min;
		else if (x > max) return max;
		else return x;
	}

	static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif
