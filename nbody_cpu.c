#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <omp.h>

#define SOFTENING 1e-9f

int nthreads = 1;

typedef struct
{
  float x, y, z, vx, vy, vz;
} Body;

void particle_positions_to_csv(FILE *datafile, int iter, Body *p, int nBodies)
{
  for (int i = 0; i < nBodies; i++)
  {
    fprintf(datafile, "%i, %f, %f, %f\n", iter, p[i].x, p[i].y, p[i].z);
  }
}

void randomizeBodies(float *data, int n)
{
#ifdef OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
  for (int i = 0; i < n; i++)
  {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n)
{

#ifdef OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
  for (int i = 0; i < n; i++)
  {
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j = 0; j < n; j++)
    {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

int main(const int argc, const char **argv)
{
  int nBodies = 3000;

  if (argc > 1)
    nBodies = atoi(argv[1]);
  if (argc > 2)
    nthreads = atoi(argv[2]);

  const float dt = 0.01f; // time step
  const int nIters = 20;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf = (float *)malloc(bytes);
  Body *p = (Body *)buf;

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

  double totalTime = 0.0;

#ifdef WRITETOFILE
  FILE *datafile;
  datafile = fopen("nbody.csv", "w");
#endif
  // fprintf(datafile,"%d %d %d\n", nBodies, nIters, 0);

  /* ------------------------------*/
  /*     MAIN LOOP                 */
  /* ------------------------------*/
  for (int iter = 1; iter <= nIters; iter++)
  {
    // printf("iteration:%d\n", iter);

#ifdef WRITETOFILE
    particle_positions_to_csv(datafile, iter, p, nBodies);
#endif

    StartTimer();

    bodyForce(p, dt, nBodies); // compute interbody forces

#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nBodies; i++)
    { // integrate position
      p[i].x += p[i].vx * dt;
      p[i].y += p[i].vy * dt;
      p[i].z += p[i].vz * dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1)
    { // First iter is warm up
      totalTime += tElapsed;
    }
  }

#ifdef WRITETOFILE
  fclose(datafile);
#endif
  double avgTime = totalTime / (double)(nIters - 1);

  printf("avgTime: %f   totTime: %f \n", avgTime, totalTime);
  free(buf);
}
