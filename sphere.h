#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

/*EXPLANATION:
equation of a sphere: (x - cx)^2 + (y - cy)^2 + (z - cz)^2 = r^2
using vec3, we get (P - C) where P is a point on the sphere and C is the center of the sphere
converting to xyz, we get (px-cx, py-cy, pz-cz)
dot product of (P-C) . (P-C) gives (px-cx)*(px-cx) + (py-cy)*(py-cy) + (pz-cz)*(pz-cz)
simplifying, we get (x - cx)^2 + (y - cy)^2 + (z - cz)^2 = r^2 again
therefore, (P-C) . (P-C) = r^2
------------------------------
We want to know if our ray ever satisfies this equation
a ray, P(t) = A + tB where A is its origin, B is the direction, and t is some distance along the ray
Therefore, to solve the equation above, given an origin and direction for our rays, we must find if any t intersects with the sphere
(tB + A - C) . (tB + A - C) = r^2
going through with the dot products, and moving the r^2, we get
t^2 B.B + 2tB . (A-C) + (A-C) . (A-C) - r^2 = 0
again, A, B, C and r are all known quantities. t is the only unknown, thus this equation is quadratic
let a = b.b
let b = 2b . (A - C)
let c = (A-C) . (A-C) - r^2
a(t^2) + tb + c
(-b +/- sqrt(b^2 - 4ac) / 2a)
however, b has a factor of 2, which comes out of the sqrt to simplifiy the equation
(-b +/- sqrt(b^2 -ac))/a

additionally, a vector dotted with itself equals the squared length of that vector, therefore a can be simplified to just get the length of the ray, no dot product required
------------------------------
using the discriminant, we can find how many solutions there are to this equation
a negative discriminant gives an imaginary solution
a zero discriminant gives one solution (tangent, one point where the ray intersects)
a positive discriminant gives two solutions (the ray intersects the sphere in one place and exits it in another)

this returns the t along the hit_record (see hittable.h)
*/

class sphere : public hittable {
public:
	sphere(point3 _center, double _radius, shared_ptr<material> _material) : center(_center), radius(_radius), mat(_material) {}

	bool hit(const ray&r, interval ray_t, hit_record& rec) const override{
		vec3 oc = r.origin() - center; //A-C
		double a = r.direction().length_squared(); //B.B
		double half_b = dot(oc, r.direction());
		double c = oc.length_squared() - radius * radius;

		double discriminant = half_b*half_b - a*c;
		if (discriminant < 0) return false;
		double sqrtd = sqrt(discriminant);
		
		//nearest t in the accepted range, test +/-
		double root = (-half_b - sqrtd) / a;
		if(!ray_t.surrounds(root)){
			root = (-half_b + sqrtd)/a;
			if(!ray_t.surrounds(root)){
				return false; //no solutions in the range
			}
		}
	
		rec.t = root;
		rec.p = r.at(rec.t);
		vec3 outward_normal = (rec.p - center) / radius;
		rec.set_face_normal(r, outward_normal);
		rec.mat = mat;
		
		return true;
	}
	
private:
	point3 center;
	double radius;
	shared_ptr<material> mat;
};

#endif
