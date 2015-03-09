#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

#include <cuda_runtime.h>
#include "cuSPH.h"

namespace licht
{
	namespace cuSPH3D
	{
		struct Particle : public Particle3DBase{
		};
		struct ParticleTmp : public Particle3DTmpBase{
		};
		class cuSPH3D
			: public cuSPH3DBase<struct Particle, struct ParticleTmp,16U,16U,16U>
		{
			private:
				SolidSphere sphere;
			public:
				cuSPH3D(unsigned int n, double time_int, struct Box box)
					: cuSPH3DBase(n, time_int, box), sphere(0.1,10,10)
				{
				}
				~cuSPH3D()
				{ }

				virtual void displayFunc()
				{
					auto& p = particles.getFromDevice();
					for(int i=0; i<particles.size; i++)
					{
						auto x = p[i].p.x,
						     y = p[i].p.y,
						     z = p[i].p.z;
						std::vector<GLfloat> color = {0.0,1.0,0,1.0};
						sphere.draw(x,y,z,color);
					}
				}

				void pInit()
				{
					auto& p = particles.getHostPtr();
					auto size = particles.size;

					for(int i=0; i<size; i++)
					{
						p[i].p.x = 10.0*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
						p[i].p.y = 10.0*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
						p[i].p.z = 10.0*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
						// p[i].v.x = 10.0*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
						// p[i].v.y = 10.0*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
						// p[i].v.z = 10.0*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
						// p[i].v.x = 0;
						// p[i].v.y = 0;
						// p[i].v.z = 0;
						p[i].v.x = 0;
						p[i].v.y = 10;
						p[i].v.z = 0;


						p[i].r = 0.1;

						p[i].m = 0.01;
					}
					particles.setToDevice();
				}

				void debug()
				{
					auto& p = particles.getFromDevice();
					auto size = particles.size;

					for(int i=0; i<size; i++)
					{
						std::cout
							<< p[i].p.z
							<< std::endl;
					}
				}
		};

		template <class T, class U, unsigned int X, unsigned int Y, unsigned int Z>
			__global__ void kineticEffects(T* p, U* tmp, struct Bucket (*buckets_d)[X][Y], struct Box* box_d)
			{
				const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

				auto& particle = p[id];
				auto& particle_tmp = tmp[id];

				particle.f.x = - 0.01*particle.v.x;
				particle.f.y = - 0.01*particle.v.y;
				particle.f.z = -9.8*particle.m - 0.01*particle.v.z;

				double xmax = box_d->x;
				double ymax = box_d->y;
				double zmax = box_d->z;

				auto xloc = particle.bucket_loc.x;
				auto yloc = particle.bucket_loc.y;
				auto zloc = particle.bucket_loc.z;

				auto& bucket = buckets_d[xloc][yloc][zloc];
				for(int i=0; i<bucket.count; i++)
				{
					if(i==particle.bucket_loc.no)
					{
						continue;
					}
					auto& particle_near = p[bucket.list[i]];
					double d = dist(particle.p, particle_near.p);
					double R = particle.r+particle_near.r;
					if(d<R)
					{
						auto v = particle.p-particle_near.p;
						particle.f
							= ( (100*pow(d-R, 2.0))*v )/abs(v);
					}
				}

				if(zloc<=1)
				{
					if(particle.p.z<particle.r)
					{
						particle.f.z += 10000*pow(particle.p.z-particle.r,2.0);
					}
				}
				else if(zloc>=Z-2)
				{
					if(zmax-particle.p.z<particle.r)
					{
						particle.f.z -= 10000*pow(zmax-particle.p.z-particle.r,2.0);
					}
				}

				if(xloc<=1)
				{
					if(particle.p.x<particle.r)
					{
						particle.f.x += 10000*pow(particle.p.x-particle.r,2.0);
					}
				}
				else if(xloc>=X-2)
				{
					if(xmax-particle.p.x<particle.r)
					{
						particle.f.x -= 10000*pow(xmax-particle.p.x-particle.r,2.0);
					}
				}

				if(yloc<=1)
				{
					if(particle.p.y<particle.r)
					{
						particle.f.y += 10000*pow(particle.p.y-particle.r,2.0);
					}
				}
				else if(yloc>=Y-2)
				{
					if(ymax-particle.p.y<particle.r)
					{
						particle.f.y -= 10000*pow(ymax-particle.p.y-particle.r,2.0);
					}
				}

			}

		template <class T, class U>
			__global__ void swap(T* p, U* tmp)
			{
				const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

				auto& particle = p[id];
				auto& particle_tmp = tmp[id];

				particle.p = particle_tmp.p;
				particle.v = particle_tmp.v;
			}

		template <class T, unsigned int X, unsigned int Y, unsigned int Z>
			__global__ void boundary(T* p, struct Box* box_d)
			{
			}

	}
}

int main(void)
{
	struct licht::Box box;
	box.x = 10.0, box.y = 10.0, box.z = 10.0, box.rmin = 0.1;

	licht::cuSPH3D::cuSPH3D sph3d(10240, 0.001, box);
	sph3d.displayCreate(std::string("sph3d"));
	sph3d.pInit();
	//sph3d.sort();
	for(unsigned long long int t=1LLU; t<=1000000LLU; t++){
		sph3d.sort();
		sph3d.move();
		sph3d.display();
	}
	std::this_thread::sleep_for(std::chrono::seconds(10));
}
