nbody_cpu: timer.h timer.c nbody_cpu.c
	gcc -O3 -flto -march=native -mtune=native -o nbody_cpu timer.c nbody_cpu.c -lm

nbody_cpu_parallel: timer.h timer.c nbody_cpu.c
	gcc -O3 -flto -fopenmp -DOPENMP -march=native -mtune=native -o nbody_cpu_parallel timer.c nbody_cpu.c -lm

cpu_plot: timer.h timer.c nbody_cpu.c
	gcc -O3 -flto -fopenmp -DOPENMP -DWRITETOFILE -march=native -mtune=native -o cpu_plot timer.c nbody_cpu.c -lm

nbody_gpu: timer.h timer.cu nbody_gpu.cu 
	nvcc -arch sm_70 -o nbody_gpu timer.cu nbody_gpu.cu -Xcompiler -lm -Xcompiler -Wall

gpu_plot: timer.h timer.cu nbody_gpu.cu plot.py
	nvcc -arch sm_70 -DWRITETOFILE -o gpu_plot timer.cu nbody_gpu.cu -Xcompiler -lm -Xcompiler -Wall
