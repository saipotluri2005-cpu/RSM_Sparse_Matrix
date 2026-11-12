// omp_barrier_cost.cpp -- cost of OpenMP synchronization at the current
// thread count and placement (OMP_NUM_THREADS, OMP_PLACES, OMP_PROC_BIND).
//
//   omp_barrier_cost [REPS]
//
// Prints one CSV line: threads,proc_bind,barrier_ns,reduction_ns
//   barrier_ns    one "#pragma omp barrier"
//   reduction_ns  one "#pragma omp for reduction(+:s)" over one item per
//                 thread (combine + implicit barrier), the pattern
//                 cpu_sparse uses three times per CG iteration
//
// Generated with assistance of claude code

#include <omp.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
    const long reps = argc > 1 ? std::atol(argv[1]) : 100000;
    int team = 1;
    double t_bar = 0.0, t_red = 0.0, s = 0.0; // s: reduction target, shared

#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();

        for (long r = 0; r < 1000; ++r) // warm up
        {
#pragma omp barrier
        }
        double t0 = omp_get_wtime();
        for (long r = 0; r < reps; ++r)
        {
#pragma omp barrier
        }
        double t1 = omp_get_wtime();
#pragma omp master
        t_bar = t1 - t0;

#pragma omp barrier
        t0 = omp_get_wtime();
        for (long r = 0; r < reps; ++r)
        {
#pragma omp for reduction(+ : s) schedule(static)
            for (int i = 0; i < team; ++i) s += 1.0;
        }
        t1 = omp_get_wtime();
#pragma omp master
        t_red = t1 - t0;
    }
    const char *bind = std::getenv("OMP_PROC_BIND");
    std::printf("threads,proc_bind,barrier_ns,reduction_ns\n%d,%s,%.1f,%.1f\n", team, bind ? bind : "unset",
                t_bar / reps * 1e9, t_red / reps * 1e9);
    return s < 0; // keep the reduction live
}
