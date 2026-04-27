# GPU Computing Assignments (UCS635)

## Student Details
- **Name:** Suryansh Sharma
- **Roll No:** 102316055
- **Batch:** 3P12

This repository contains solved programs for all provided assignments:
- OpenMP Assignment 1-2
- Lab Assignment 3-4
- Lab Assignment 5
- Lab Assignment 6-7
- Lab Assignment 8-11

## Folder Structure
- `solutions/assignment_1_2/` → OpenMP basics (threads, private, lastprivate)
- `solutions/assignment_3_4/` → OpenMP scheduling, sections, synchronization, reduction
- `solutions/assignment_5/` → Random dataset generation + CUDA env/device checks
- `solutions/assignment_6_7/` → CUDA matrix add/mul, CURAND timing, thread scaling
- `solutions/assignment_8_11/` → CUDA Q1-Q8 kernels (tiling, reduction, sorting, convolution)

## Build and Run (Linux)

### OpenMP (GCC)
```bash
gcc -fopenmp -O2 solutions/assignment_1_2/family_members_threads.c -o family
./family
```

Example for matrix schedule compare:
```bash
gcc -fopenmp -O2 solutions/assignment_3_4/matrix_mul_schedule_compare.c -o mm_sched
./mm_sched 512
```

### CUDA (NVCC)
```bash
nvcc -O2 solutions/assignment_6_7/cuda_matrix_ops_timing.cu -o cuda_mat_ops
./cuda_mat_ops 1024
```

CURAND program build example:
```bash
nvcc -O2 solutions/assignment_6_7/curand_generation_timing.cu -lcurand -o curand_time
./curand_time 1048576
```

### Python Scripts (Assignment 5)
```bash
python3 solutions/assignment_5/random_dataset_timing.py --n 1000000 --outdir datasets
python3 solutions/assignment_5/numba_cupy_check.py
```

## Git Setup and Push (Exact Commands)
Run these from repository root:

```bash
git init
git add .
git commit -m "Add complete GPU computing assignment solutions"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

If Git asks for identity:

```bash
git config --global user.name "Suryansh Sharma"
git config --global user.email "your-email@example.com"
```

## Notes
- Some CUDA files need an NVIDIA GPU + CUDA toolkit.
- For very large sizes (like 10000x10000), ensure enough GPU memory.
- Timing values will vary by hardware (laptop vs colab CPU/GPU/TPU).
