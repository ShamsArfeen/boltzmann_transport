#include <math.h>
#include <stdio.h>

double bessel_Kn(int n, double z)
{
    if (z <= 0.0 || n < 0) return NAN;

    /* integration parameters */
    const int N = 4000;        // must be even
    const double t_max = 10.0; // sufficient for z >= O(0.1)
    const double h = t_max / N;

    double sum = 0.0;

    for (int i = 0; i <= N; i++) {
        double t = i * h;
        double f = exp(-z * cosh(t)) * cosh(n * t);

        if (i == 0 || i == N)
            sum += f;
        else if (i % 2 == 0)
            sum += 2.0 * f;
        else
            sum += 4.0 * f;
    }

    return (h / 3.0) * sum;
}
