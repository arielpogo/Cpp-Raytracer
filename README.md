# Cpp-Raytracer
- Created with the help of the "Raytracing in One Weekend" trilogy of books: https://raytracing.github.io/
- I plan to implement parallelization with CUDA in the future
- Outputs in the PPM format for now. To view: https://0xc0de.fr/webppm/
- Default scene for now includes 3 spheres, a giant grey sphere, a small yellow sphere and a small reflective golden sphere

### To compile (with g++):
g++ main.cpp -o rt -Wall -O3

### To run:
- linux: ./rt.out (args)
- windows: rt.exe (args)

### Args:
- -i: info on args
- -d: enable debug
- -o <name>: specify output file name
- -h <int>: specifiy image height
- -r <int> specify preset ratio (4 = 4:3, 16 = 16:9, 10 = 16:10, 1 = 1:1)
