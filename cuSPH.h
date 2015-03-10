#pragma once

#include <type_traits>

#include <vector>
#include <limits>
#include <string>
#include <memory>
#include <cmath>

#include <cuda_runtime.h>

#ifdef __APPLE__

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#else

#include <GL/gl.h>
#include <GL/glu.h>

#endif
#include <GLFW/glfw3.h>

namespace licht
{
	struct Bucket
	{
		unsigned int count;
		unsigned int* list;
	};

	struct Box
	{
		double x;
		double y;
		double z;

		double rmin;
	};

	template <class A, class U>
		class cuSPHParticles
		{
			public:
				dim3 block;
				dim3 grid;
				A* particles_d;
				U* particles_d_tmp;
				std::unique_ptr<A[]> particles_h;
				unsigned int size;

				cuSPHParticles(unsigned int n)
					: grid(n/512U,1,1), block(512,1,1), particles_h(new A[n])
				{
					// particles_h = std::make_shared<A[n]>();
					size = n;
					cudaMalloc(
							reinterpret_cast<void**>(&particles_d),
							n*sizeof(A)
						  );
					cudaMalloc(
							reinterpret_cast<void**>(&particles_d_tmp),
							n*sizeof(U)
						  );
				}

				~cuSPHParticles()
				{
					cudaFree(reinterpret_cast<void*>(particles_d));
					cudaFree(reinterpret_cast<void*>(particles_d_tmp));
				}

				std::unique_ptr<A[]>& getFromDevice(void)
				{
					cudaMemcpy(
							particles_h.get(),
							particles_d,
							size*sizeof(A),
							cudaMemcpyDeviceToHost
						  );
					return particles_h;
				}

				void setToDevice(void)
				{
					cudaMemcpy(
							particles_d,
							particles_h.get(),
							size*sizeof(A),
							cudaMemcpyHostToDevice
						  );
				}

				std::unique_ptr<A[]>& getHostPtr(void)
				{
					return particles_h;
				}
		};

	namespace cuSPH2D
	{
		struct BucketLoc2D
		{
			unsigned int no;
			unsigned int x;
			unsigned int y;
		};
		struct Particle2DBase
		{
			double x;
			double y;
			double r;
			struct BucketLoc2D bucket_loc;
		};
		struct Particle2DTmpBase
		{
			double x;
			double y;
		};

		template <class A, class U, unsigned int X>
			__global__ void kineticEffects(A* p, U* tmp, struct Bucket (*buckets_d)[X], struct Box* box_d);
		template <class A, class U>
			__global__ void swap(A* p, U* tmp);
		template <class A>
			__global__ void boundary(A* p, struct Box* box_d);

		template <unsigned int X>
			__global__ void bucketsAlloc(struct Bucket (*buckets_d)[X], unsigned int k)
			{
				const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

				buckets_d[x][y].count = 0U;
				buckets_d[x][y].list
					= reinterpret_cast<unsigned int*>(malloc(k*sizeof(unsigned int)));
			}
		template <unsigned int X>
			__global__ void bucketsFree(struct Bucket (*buckets_d)[X])
			{
				const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

				free(reinterpret_cast<void*>(buckets_d[x][y].list));
			}
		template <unsigned int X>
			__global__ void bucketsReset(struct Bucket (*buckets_d)[X])
			{
				const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

				buckets_d[x][y].count = 0U;
			}

		template <class A, unsigned int X, unsigned int Y>
			__global__ static void bucketsSort(A* particles_d, struct Bucket (*buckets_d)[X], struct Box* box_d)
			{
				const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

				const unsigned int bucket_max = max(X, Y);

				A& particle = particles_d[id];

				auto px = particle.x,
				     py = particle.y;

				auto xmax = box_d->x,
				     ymax = box_d->y;

				auto dx = xmax/(X-2),
				     dy = ymax/(Y-2);

				unsigned int x,y;
				bool xflag=true, yflag=true;
				double xx=0.0, yy=0.0;
				if(px<xx)
				{
					x=0U;
					xflag=false;
				}
				if(py<yy)
				{
					y=0U;
					yflag=false;
				}

				for (unsigned int i=1; i<bucket_max-1U; i++)
				{
					xx+=dx, yy+=dy;
					if(xflag && px<=xx)
					{
						x=i;
						xflag=false;
					}
					if(yflag && py<=yy)
					{
						y=i;
						yflag=false;
					}
				}

				if(px>xmax)
				{
					x=X-1U;
				}
				if(py>ymax)
				{
					y=Y-1U;
				}
				particle.bucket_loc.x = x;
				particle.bucket_loc.y = y;
				particle.bucket_loc.no = atomicAdd(&buckets_d[x][y].count,1U);
				buckets_d[x][y].list[particle.bucket_loc.no]=id;
			}
	}

	namespace cuSPH3D
	{
		template<typename A> struct Vec3{
			A x;
			A y;
			A z;
		};

		template<typename A> Vec3<A> __device__ __host__ operator +(const Vec3<A>& lhs, const Vec3<A>& rhs)
		{
			Vec3<A> ret;
			ret.x = lhs.x+rhs.x;
			ret.y = lhs.y+rhs.y;
			ret.z = lhs.z+rhs.z;
			return ret;
		}

		template<typename A> Vec3<A> __device__ __host__ operator -(const Vec3<A>& lhs, const Vec3<A>& rhs)
		{
			Vec3<A> ret;
			ret.x = lhs.x-rhs.x;
			ret.y = lhs.y-rhs.y;
			ret.z = lhs.z-rhs.z;
			return ret;
		}


		template<typename A, typename T> __device__ __host__ Vec3<A> operator *(const T& lhs, const Vec3<A>& rhs)
		{
			Vec3<A> ret;
			ret.x = lhs*rhs.x;
			ret.y = lhs*rhs.y;
			ret.z = lhs*rhs.z;
			return ret;
		}

		template<typename A, typename T> __device__ __host__ Vec3<A> operator /(const Vec3<A>& lhs, const T& rhs)
		{
			Vec3<A> ret;
			ret.x = lhs.x/rhs;
			ret.y = lhs.y/rhs;
			ret.z = lhs.z/rhs;
			return ret;
		}

		template <typename T> __device__ __host__ typename std::enable_if<std::is_floating_point<T>::value, T>::type dist(Vec3<T>& lhs, Vec3<T>& rhs)
		{
			return sqrt(
					pow(lhs.x-rhs.x,2.0)
					+ pow(lhs.y-rhs.y,2.0)
					+ pow(lhs.z-rhs.z,2.0)
				   );
		}

		template <typename T> __device__ __host__ typename std::enable_if<std::is_floating_point<T>::value, T>::type abs(Vec3<T>& v)
		{
			return sqrt( pow(v.x,2.0)+pow(v.y,2.0)+pow(v.z,2.0) );
		}

		struct BucketLoc3D
		{
			unsigned int no;
			unsigned int x;
			unsigned int y;
			unsigned int z;
		};
		struct Particle3DBase
		{
			double r;
			double m;
			struct Vec3<double> p;
			struct Vec3<double> v;
			struct Vec3<double> f;
			struct BucketLoc3D bucket_loc;
		};
		struct Particle3DTmpBase
		{
			struct Vec3<double> p;
			struct Vec3<double> v;
		};

		template <class A, class U, unsigned int X, unsigned int Y, unsigned int Z>
			__global__ void kineticEffects(A* p, U* tmp, struct Bucket (*buckets_d)[X][Y], struct Box* box_d);
		template <class A, class U>
			__global__ void swap(A* p, U* tmp);
		template <class A, unsigned int X, unsigned int Y, unsigned int Z>
			__global__ void boundary(A* p, struct Box* box_d);

		template<class A> __device__ Vec3<A> euler(const Vec3<A>& v, const Vec3<A>& f, const double h)
		{
			return v + h*f;
		}

		template <class A, class B> __global__ void Euler(A* particles_d, B* particles_d_tmp, double t)
		{
			const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

			A& p = particles_d[id];
			B& pt = particles_d_tmp[id];

			pt.v = euler(p.v, p.f/p.m, t);
			pt.p = euler(p.p, pt.v, t);
		}

		template <unsigned int X, unsigned int Y>
			__global__ void bucketsAlloc(struct Bucket (*buckets_d)[X][Y], unsigned int k)
			{
				const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
				const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

				buckets_d[x][y][z].count = 0U;
				buckets_d[x][y][z].list
					= reinterpret_cast<unsigned int*>(malloc(k*sizeof(unsigned int)));
			}
		template <unsigned int X, unsigned int Y>
			__global__ void bucketsFree(struct Bucket (*buckets_d)[X][Y])
			{
				const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
				const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

				free(reinterpret_cast<void*>(buckets_d[x][y][z].list));
			}
		template <unsigned int X, unsigned int Y>
			__global__ void bucketsReset(struct Bucket (*buckets_d)[X][Y])
			{
				const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
				const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

				buckets_d[x][y][z].count = 0U;
			}

		template <class A, unsigned int X, unsigned int Y, unsigned int Z>
			__global__ static void bucketsSort(A* particles_d, struct Bucket (*buckets_d)[X][Y], struct Box* box_d)
			{
				const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

				const unsigned int bucket_max = max(max(X, Y), Z);

				A& particle = particles_d[id];

				auto px = particle.p.x,
				     py = particle.p.y,
				     pz = particle.p.z;

				auto xmax = box_d->x,
				     ymax = box_d->y,
				     zmax = box_d->z;

				auto dx = xmax/(X-2),
				     dy = ymax/(Y-2),
				     dz = zmax/(Z-2);

				unsigned int x,y,z;

				if(px<0.0)
				{
					x=0U;
				}
				else if(px>xmax)
				{
					x=X-1U;
				}
				else
				{
					x=1U+static_cast<unsigned int>(px/dx);
				}

				if(py<0.0)
				{
					y=0U;
				}
				else if(py>ymax)
				{
					y=Y-1U;
				}
				else
				{
					y=1U+static_cast<unsigned int>(py/dy);
				}

				if(pz<0.0)
				{
					z=0U;
				}
				else if(pz>zmax)
				{
					z=Z-1U;
				}
				else
				{
					z=1U+static_cast<unsigned int>(pz/dz);
				}

				// unsigned int x,y,z;
				// bool xflag=true, yflag=true, zflag=true;
				// double xx=0.0, yy=0.0, zz=0.0;
				// if(px<xx)
				// {
				// 	x=0U;
				// 	xflag=false;
				// }
				// if(py<yy)
				// {
				// 	y=0U;
				// 	yflag=false;
				// }
				// if(pz<zz)
				// {
				// 	z=0U;
				// 	zflag=false;
				// }
				//
				// for (unsigned int i=1; i<bucket_max-1U; i++)
				// {
				// 	xx+=dx, yy+=dy, zz+=dz;
				// 	if(xflag && px<=xx)
				// 	{
				// 		x=i;
				// 		xflag=false;
				// 	}
				// 	if(yflag && py<=yy)
				// 	{
				// 		y=i;
				// 		yflag=false;
				// 	}
				// 	if(zflag && pz<=zz)
				// 	{
				// 		z=i;
				// 		zflag=false;
				// 	}
				// }
				//
				// if(px>xmax)
				// {
				// 	x=X-1U;
				// }
				// if(py>ymax)
				// {
				// 	y=Y-1U;
				// }
				// if(pz>zmax)
				// {
				// 	z=Z-1U;
				// }

				particle.bucket_loc.x = x;
				particle.bucket_loc.y = y;
				particle.bucket_loc.z = z;
				particle.bucket_loc.no = atomicAdd(&buckets_d[x][y][z].count,1U);

				// printf("%u, %u, %u, %u\n",particle.bucket_loc.no,x,y,z);

				buckets_d[x][y][z].list[particle.bucket_loc.no]=id;
			}

		template <unsigned int X, unsigned int Y, unsigned int Z>
			class cuSPH3DMap
			{
				public:
					dim3 block;
					dim3 grid;
					struct Box box_h;
					struct Box* box_d;
					struct Bucket (*buckets_d)[X][Y];

					cuSPH3DMap(unsigned int n, struct Box box)
						: grid(X/8U, Y/8U, Z/8U), block(8, 8, 8)
					{
						box_h = box;
						cudaMalloc(
								reinterpret_cast<void**>(&box_d),
								sizeof(struct Box)
							  );
						cudaMemcpy(
								reinterpret_cast<void*>(box_d),
								reinterpret_cast<void*>(&box_h),
								sizeof(struct Box),
								cudaMemcpyHostToDevice
							  );

						cudaMalloc(
								reinterpret_cast<void**>(&buckets_d),
								X*Y*Z*sizeof(struct Bucket)
							  );

						auto dx = (box.x)/(X-2U);
						auto dy = (box.y)/(Y-2U);
						auto dz = (box.z)/(Z-2U);

						auto vf = dx * dy * dz * 0.8;
						auto vb = 4.0 * M_PI * std::pow(box.rmin, 3.0) / 3.0;
						unsigned int k = vf/vb;
						k++;
						bucketsAlloc<<<grid,block>>>(buckets_d,k);
					}

					~cuSPH3DMap()
					{
						cudaFree(reinterpret_cast<void*>(box_d));
						cudaFree(reinterpret_cast<void*>(buckets_d));
						bucketsFree<<<grid,block>>>(buckets_d);
					}
			};

		class SolidSphere
		{
			protected:
				std::vector<GLfloat> vertices;
				std::vector<GLfloat> normals;
				std::vector<GLfloat> texcoords;
				std::vector<GLushort> indices;

			public:
				SolidSphere(float radius, unsigned int rings, unsigned int sectors)
				{
					const float R = 1./(float)(rings-1);
					const float S = 1./(float)(sectors-1);

					vertices.resize(rings * sectors * 3);
					normals.resize(rings * sectors * 3);
					texcoords.resize(rings * sectors * 2);

					auto v = vertices.begin();
					auto n = normals.begin();
					auto t = texcoords.begin();
					for(int r = 0; r < rings; r++)
					{
						for(int s = 0; s < sectors; s++)
						{
							float const y = sin( -M_PI_2 + M_PI * r * R );
							float const x = cos(2*M_PI * s * S) * sin( M_PI * r * R );
							float const z = sin(2*M_PI * s * S) * sin( M_PI * r * R );

							*t++ = s*S;
							*t++ = r*R;

							*v++ = x * radius;
							*v++ = y * radius;
							*v++ = z * radius;

							*n++ = x;
							*n++ = y;
							*n++ = z;

						}
					}

					indices.resize(rings * sectors * 4);
					auto i = indices.begin();
					for(int r = 0; r < rings-1; r++)
					{
						for(int s = 0; s < sectors-1; s++)
						{
							*i++ = r * sectors + s;
							*i++ = r * sectors + (s+1);
							*i++ = (r+1) * sectors + (s+1);
							*i++ = (r+1) * sectors + s;
						}
					}
				}

				void draw(GLfloat x, GLfloat y, GLfloat z, std::vector<GLfloat> color)
				{
					glMatrixMode(GL_MODELVIEW);
					glPushMatrix();
					// glLoadIdentity();

					glTranslatef(x,y,z);

					glDisable(GL_TEXTURE_2D);
					glEnable(GL_LIGHTING);
					glEnable(GL_LINE_SMOOTH);
					glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
					glEnable(GL_BLEND);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

					glEnableClientState(GL_VERTEX_ARRAY);
					glEnableClientState(GL_NORMAL_ARRAY);
					glEnableClientState(GL_TEXTURE_COORD_ARRAY);

					glMaterialfv(GL_FRONT_AND_BACK , GL_DIFFUSE , color.data());

					glVertexPointer(3, GL_FLOAT, 0, vertices.data());
					glNormalPointer(GL_FLOAT, 0, normals.data());
					glTexCoordPointer(2, GL_FLOAT, 0, texcoords.data());

					glDrawElements(GL_QUADS, indices.size(), GL_UNSIGNED_SHORT, indices.data());

					glDisableClientState(GL_VERTEX_ARRAY);
					glDisableClientState(GL_NORMAL_ARRAY);
					glDisableClientState(GL_TEXTURE_COORD_ARRAY);

					glPopMatrix();
				}
		};

		class cuSPH3DDisplay
		{
			private:
				struct Box box;
				GLFWwindow* window;

				static void error_callback(int eror, const char* description)
				{
					fputs(description, stderr);
				}
				void displayInit()
				{
					glfwSetErrorCallback(error_callback);
					glfwInit();
				}
				void displayTerminate()
				{
					glfwTerminate();
				}
				void displayDestroy()
				{
					glfwSetWindowShouldClose(window, GL_TRUE);
					glfwDestroyWindow(window);
				}
				void reshape(int w, int h){
					glViewport(0, 0, w, h);

					glMatrixMode(GL_PROJECTION);
					glLoadIdentity();
					double aspect =
						static_cast<double>(w)/static_cast<double>(h);
					gluPerspective(60.0, aspect, 1.0, 100.0);
					glMatrixMode(GL_MODELVIEW);
				}


			protected:
				cuSPH3DDisplay(struct Box b)
				{
					box = b;
					window =nullptr;
					displayInit();
				}
				virtual ~cuSPH3DDisplay()
				{
					if(!(window==nullptr))
					{
						displayDestroy();
					}
					displayTerminate();
				}

			public:
				virtual void displayFunc()=0;
				void displayCreate(std::string str)
				{
					window = glfwCreateWindow(800, 800, str.c_str(), NULL, NULL);
					//glfwSetKeyCallback(window, key_callback);
					glfwMakeContextCurrent(window);
					glClearColor(0.0, 0.0, 0.0, 1.0);
					glEnable(GL_LIGHTING);
					glEnable(GL_LIGHT0);
					glEnable(GL_NORMALIZE);
					glEnable(GL_DEPTH_TEST);
					glDepthFunc(GL_LEQUAL);
					glDisable(GL_CULL_FACE);
					glCullFace(GL_BACK);
				}
				void display(void)
				{
					int w,h;
					glfwGetFramebufferSize(window, &w, &h);
					reshape(w,h);

					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					glLoadIdentity();
					gluLookAt(
							2*box.x, 2*box.y, 2*box.z,
							box.x/2, box.y/2, box.z/2,
							0.0, 0.0, 1.0
						 );
					// gluLookAt(
					// 		10.0, 10.0, 10.0,
					// 		0.0, 0.0, 0.0,
					// 		0.0, 1.0, 0.0
					// 	 );

					static float Light0Pos[]={0,0,0,0};
					Light0Pos[0] = box.x/2;
					Light0Pos[2] = -box.y/2;
					Light0Pos[2] = box.z/2;
					glLightfv(GL_LIGHT0, GL_POSITION, Light0Pos);

					//Draw
					displayFunc();

					glfwSwapBuffers(window);
				}

		};

		template<class A, class U, unsigned int X, unsigned int Y, unsigned int Z>
			class cuSPH3DBase : public cuSPH3DDisplay
		{
			protected:
				cuSPH3DMap<X, Y, Z> map;
				cuSPHParticles<A, U> particles;
				double time_interval;

				cuSPH3DBase(unsigned int n, double time_int, struct Box box)
					: map(n,box), particles(n), cuSPH3DDisplay(box)
				{
					time_interval = time_int;
				}
				virtual ~cuSPH3DBase()
				{
				}

			public:
				virtual void displayFunc()=0;

				void sort()
				{
					bucketsSort<A,X,Y,Z>
						<<<particles.grid, particles.block>>>
						(particles.particles_d, map.buckets_d, map.box_d);
				}

				void move()
				{
					kineticEffects<A,U,X,Y,Z>
						<<<particles.grid, particles.block>>>
						(particles.particles_d, particles.particles_d_tmp, map.buckets_d, map.box_d);
					Euler
						<<<particles.grid, particles.block>>>
						(particles.particles_d, particles.particles_d_tmp, time_interval);

					swap
						<<<particles.grid, particles.block>>>
						(particles.particles_d, particles.particles_d_tmp);
					boundary<A,X,Y,Z>
						<<<particles.grid, particles.block>>>
						(particles.particles_d, map.box_d);
					bucketsReset
						<<<map.grid,map.block>>>
						(map.buckets_d);
				}
		};
	}
}
