#ifndef INTERVAL_H
#define INTERVAL_H

class interval{
public:
	double min, max;

	interval() : min(+infinity), max(-infinity) {} //empty on default
	interval(double _min, double _max) : min(_min), max(_max) {}

	//whether x is in range
	bool contains(double x) const {
		return min <= x && x <= max;
	}
	
	//whether x is in range, exclusive
	bool surrounds(double x) const {
		return min < x && x < max;
	}

	double clamp(double x) const{
		if(x < min) return min;
		else if (x > max) return max;
		else return x;
	}

	static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif
