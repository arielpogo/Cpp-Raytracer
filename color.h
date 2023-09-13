#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

using color = vec3;

//images are usually gamma corrected and image viewers anticipate gamma-corrected photos
//thus, to get image viewers to more accurately display our image, we do a transform
//a simple and common one is gamma 2, where gamma space to linear you power by 2
//the inverse of powering by 2 is sqrt, thus linear to gamma:
//(the colors are a double from 0-1 so sqrt actually increases it)
inline double linear_to_gamma(double l){ return sqrt(l); }

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
	
	double r = pixel_color.x();
	double g = pixel_color.y();
	double b = pixel_color.z();

	//divide color by number of samples
	double scale = 1.0 / samples_per_pixel;
	r*=scale;
	g*=scale;
	b*=scale;
	
	r = linear_to_gamma(r);
	g = linear_to_gamma(g);
	b = linear_to_gamma(b);
	
	static const interval intensity(0.000, 0.999);
	out << static_cast<int>(256 * intensity.clamp(r)) << ' '
            << static_cast<int>(256 * intensity.clamp(g)) << ' '
	    << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}

#endif
