#pragma once
#include "color.h"

class color_255 {
public:
	char e[3];

	color_255() : e{0,0,0} {

	}

	color_255(char r, char g, char b) : e{r, g, b} {

	}

	color_255(color& c, int samples) {
		double r = c.x();
		double g = c.y();
		double b = c.z();

		//divide color by number of samples
		double scale = 1.0 / samples;
		r *= scale;
		g *= scale;
		b *= scale;

		r = sqrt(r);
		g = sqrt(g);
		b = sqrt(b);

		static const interval intensity(0.000, 0.999);
		e[0] = static_cast<char>(256 * intensity.clamp(r));
		e[1] = static_cast<char>(256 * intensity.clamp(g));
		e[2] = static_cast<char>(256 * intensity.clamp(b));
	}

	char x() const { return e[0]; }
	char y() const { return e[1]; }
	char z() const { return e[2]; }
};