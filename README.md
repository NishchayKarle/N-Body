## **1.1 - Serial Version**
> **100 bodies in 2 clusters with 0 initial velocity**
![100_2_clusters_0_v](/GIFS/100_2_clusters_0_v.gif)

> **1,000 bodies in 2 clusters**
![1000_2_clusters](/GIFS/1000_2_clusters.gif)

> **1,000 bodies 1000 iterations small initial velocity**
![1000_1000](/GIFS/1000_1000.gif)

* **Execution time**
    * 100,000 bodies
        * time per iteration - 55.19s
        * iterations per sec - 0.018

---
</br>

## **1.2 - Shared Memory Parallelism**

* ```100,000 bodies with 20,000 iterations``` 

|          \#cores           |    1    |   4    |   8    |  16   |  32   |  64   |  128  |  256  |
| :------------------------: | :-----: | :----: | :----: | :---: | :---: | :---: | :---: | :---: |
| **time per iteration (s)** |  55.19  | 16.43  |  8.23  | 4.14  | 2.20  | 1.55  | 1.30  | 1.21  |
|     **total time (s)**     | 1048.67 | 312.18 | 156.46 | 78.73 | 41.85 | 29.58 | 24.71 | 23.13 |

> ![](/Plots/openmp_avg.png)

> ![](/Plots/openmp_tot.png)

> ![](/Plots/openmp_combined.png)
---
</br>

## **1.3 - GPU Implementation**
</br>

**```GRIND RATE```**
|                        |  GPU  | OpenMP | Serial |
| :--------------------: | :---: | :----: | :----: |
| time per iteration (s) | 0.051 |  1.21  | 55.19  |

* best performance was with 256 threads per block for the GPU Implementation.

* Files Serial_timesteps.csv and Cuda_timesteps.csv were generated for 20 bodies and 20 iterations with the same initial positions and velocity, which shows the correctness of the CUDA code.

---
</br>

## **Compile and Run**

* Serial
    * ```make nbody_cpu```
    * ```./nbody_cpu <number_of_bodies>```
* OPENMP
    * ```make nbody_cpu_parallel```
    * ```./nbody_cpu_parallel <number_of_bodies> <num_of_threads>```
* CUDA
    * ```make nbody_gpu```
    * ```./nbody_gpu <number_of_bodies> <threads_per_block>```

---
</br>

## **Creating Animation**
* CPU code
    * ```make cpu_plot```
    * ```./cpu_plot <number_of_bodies> <num_of_threads>```
    * ```python3 plot.py```
* CUDA
    * ```make gpu_plot``` 
    * ```./gpu_plot <number_of_bodies> <threads_per_block>```
    * ```python3 plot.py```
