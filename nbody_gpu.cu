#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <assert.h>
#include "timer.h"

#define SOFTENING 1e-4f
#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

struct Body
{
    float x, y, z, vx, vy, vz;
};
typedef struct Body Body;

__host__ void particle_positions_to_csv(FILE *datafile, int iter, Body *p, int nBodies)
{
    for (int i = 0; i < nBodies; i++)
    {
        fprintf(datafile, "%i, %f, %f, %f\n", iter, p[i].x, p[i].y, p[i].z);
    }
}

__host__ void randomizeBodies(Body *data, int n)
{
    for (int i = 0; i < n / 2; i++)
    {
        data[i].x = 0.0 + (rand() / (float)RAND_MAX) * 100;
        data[i].y = 100.0 + (rand() / (float)RAND_MAX) * 100;
        data[i].z = 0.0 + (rand() / (float)RAND_MAX) * 100;

        data[i + n / 2].x = -100.0 + (rand() / (float)RAND_MAX) * 100;
        data[i + n / 2].y = 0.0 + (rand() / (float)RAND_MAX) * 100;
        data[i + n / 2].z = 0.0 + (rand() / (float)RAND_MAX) * 100;

        // data[i].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // data[i].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // data[i].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

        // data[i + n / 2].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // data[i + n / 2].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // data[i + n / 2].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

        data[i].vx = 0;
        data[i].vy = 0;
        data[i].vz = 0;

        data[i + n / 2].vx = 0;
        data[i + n / 2].vy = 0;
        data[i + n / 2].vz = 0;
    }
}
__global__ void test(Body *particles, int nBodies,
                     float Fx, float Fy, float Fz, int i)
{

    for (int j = 0; j < nBodies; j++)
    {
        float dx = particles[j].x - particles[i].x;
        float dy = particles[j].y - particles[i].y;
        float dz = particles[j].z - particles[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
}
__global__ void bodyForce(Body *particles, float dt, int nBodies)
{
    int tid0 = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid0; i < nBodies; i += blockDim.x * gridDim.x)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < nBodies; j++)
        {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        particles[i].vx += dt * Fx;
        particles[i].vy += dt * Fy;
        particles[i].vz += dt * Fz;
    }

    // INTEGRATE POSITION
    for (int i = tid0; i < nBodies; i += blockDim.x * gridDim.x)
    {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

int main(const int argc, const char **argv)
{
    int nthreads_per_block = 128, nblocks;
    int nBodies = 3000;

    if (argc > 1)
        nBodies = atoi(argv[1]);
    if (argc > 2)
        nthreads_per_block = atoi(argv[2]);

    nblocks = MIN(nBodies / nthreads_per_block + 1, MAX_BLOCKS_PER_DIM);

    Body *particles_h = (Body *)malloc(sizeof(Body) * nBodies);
    Body *particles_d;
    assert(cudaMalloc((void **)&particles_d, sizeof(Body) * nBodies) == cudaSuccess);

    randomizeBodies(particles_h, nBodies); // Init pos / vel data
    cudaMemcpy(particles_d, particles_h, sizeof(Body) * nBodies, cudaMemcpyHostToDevice);

    // TIME STEP
    const float dt = 0.01f;
    // SIMULATION ITERATIONS
    const int nIters = 20;

#ifdef WRITETOFILE
    FILE *datafile = fopen("nbody.csv", "w");
    particle_positions_to_csv(datafile, 0, particles_h, nBodies);
#endif
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    {
        // ITERATE
        for (int iter = 1; iter <= nIters; iter++)
        {
            // printf("iteration:%d\n", iter);

            // COMPUTE INTERBODY FORCES
            bodyForce<<<nblocks, nthreads_per_block>>>(particles_d, dt, nBodies);

#ifdef WRITETOFILE
            cudaMemcpy(particles_h, particles_d, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost);

            particle_positions_to_csv(datafile, iter, particles_h, nBodies);
#endif
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float totalTime = 0.0;
    cudaEventElapsedTime(&totalTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(particles_h);
    cudaFree(particles_d);

    totalTime /= (float)1000;
    double avgTime = totalTime / (float)(nIters);
    printf("avgTime: %f (s)   totTime: %f (s)\n", avgTime, totalTime);

#ifdef WRITETOFILE
    fclose(datafile);
#endif
    return EXIT_SUCCESS;
}
