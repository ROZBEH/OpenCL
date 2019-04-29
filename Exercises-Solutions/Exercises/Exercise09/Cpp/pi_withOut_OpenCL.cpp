#include "util.hpp"
#include <cstdio>
static long num_steps = 100000000;
double step;
extern double wtime();   // returns time since some fixed past point (wtime.c)


int main ()
{
    int i;
    double x, pi, sum = 0.0;


    step = 1.0/(double) num_steps;

    util::Timer timer;

    for (i=1;i<= num_steps; i++){
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }

    pi = step * sum;
    double run_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    printf("\n pi with %ld steps is %lf in %lf seconds\n", num_steps, pi, run_time);
}
