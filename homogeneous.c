#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

/* ============================================================
   Constants
   ============================================================ */

#define     SEED            777223
#define     PI              3.141592653589793
#define     TEMPERATURE     0.6
#define     MASS            0.0
#define     SIGMA_FM2       0.5
#define     FM2             25.679

#define     NPART           1000
#define     NT              50000
#define     TMAX            500
#define     DT              0.1

#define     EPS             1e-12

/* ============================================================
   Derived thermodynamics
   ============================================================ */

static inline double number_density(void) {
    if (MASS == 0.0)
        return pow(TEMPERATURE, 3) / (PI * PI);

    double integral = 0.0;
    double dt = 1e-3;
    double beta_m = MASS/(double)TEMPERATURE;
    
    for (int idx = 0; idx < 1e5; idx++) {
        double t = dt*idx + 1.0;
        integral += exp(-beta_m*t)* pow(t*t - 1, 3/2.0);
    }
    return integral*dt*pow(MASS, 4)  / (double)(TEMPERATURE* 2*PI * PI);
}

/* ============================================================
   Vector algebra
   ============================================================ */

typedef struct {
    double x, y, z;
} Vec3;

static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x+b.x, a.y+b.y, a.z+b.z};
}

static inline Vec3 vec3_scale(Vec3 a, double s) {
    return (Vec3){a.x*s, a.y*s, a.z*s};
}

static inline double vec3_dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline double vec3_norm(Vec3 a) {
    return sqrt(vec3_dot(a,a));
}

static inline Vec3 vec3_floor(Vec3 a) {
    return (Vec3){floor(a.x), floor(a.y), floor(a.z)};
}

/* ============================================================
   RNG utilities
   ============================================================ */

static inline double rng_uniform(void) {
    return rand() / (double)RAND_MAX;
}

static inline double rng_normal(void) {
    double u, v, s;
    do {
        u = 2.0 * rng_uniform() - 1.0;
        v = 2.0 * rng_uniform() - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s < EPS);
    return u * sqrt(-2.0 * log(s) / s);
}

static inline Vec3 rng_unit_sphere(void) {
    Vec3 v = {rng_normal(), rng_normal(), rng_normal()};
    return vec3_scale(v, 1.0 / vec3_norm(v));
}

/* ============================================================
   Particle data structures
   ============================================================ */

typedef struct {
    Vec3 r;
    Vec3 p;
    double m;
} Particle;

typedef struct {
    Particle *part;
    double V;
    double L;
} ParticleSystem;

/* ============================================================
   Relativistic kinematics
   ============================================================ */

static inline double particle_energy(const Particle *part) {
    return sqrt(vec3_dot(part->p, part->p) + part->m * part->m);
}

static inline Vec3 lorentz_boost(Vec3 p, double E, Vec3 beta) {
    double beta2 = vec3_dot(beta, beta);
    if (beta2 < EPS) return p;

    double gamma = 1.0 / sqrt(1.0 - beta2);
    double p_dot_b = vec3_dot(p, beta);

    return vec3_add(
        p,
        vec3_scale(beta, (gamma - 1.0) * p_dot_b / beta2 - gamma * E)
    );
}

/* ============================================================
   Initialization
   ============================================================ */

void initialize_particles(ParticleSystem *sys) {
    double n = number_density();
    sys->V = NPART / n;
    sys->L = cbrt(sys->V);

    Vec3 p_total = {0,0,0};
    double e_total = 0.0;

    for (int i = 0; i < NPART; i++) {
        sys->part[i].m = MASS;
        sys->part[i].r = vec3_scale(
            (Vec3){rng_uniform(), rng_uniform(), rng_uniform()},
            sys->L
        );

        double beta_e;
        double beta_m = MASS / (double)TEMPERATURE;
        while (1) {
            beta_e = -log(rng_uniform()*rng_uniform()*rng_uniform() + EPS);
            if (beta_e < beta_m) continue;
            else if (rng_uniform() < sqrt(beta_e*beta_e - beta_m*beta_m) / beta_e) break;
        }

        double E = beta_e * TEMPERATURE;
        double p = sqrt(E*E - MASS*MASS);

        sys->part[i].p = vec3_scale(rng_unit_sphere(), p);

        p_total = vec3_add(p_total, sys->part[i].p);
        e_total += E;
    }

    // boost to Landau rest frame
    Vec3 beta = vec3_scale(p_total, 1.0 / fmax(e_total, EPS));
    for (int i = 0; i < NPART; i++) {
        double E = particle_energy(&sys->part[i]);
        sys->part[i].p = lorentz_boost(sys->part[i].p, E, beta);
    }
}

/* ============================================================
   Free streaming
   ============================================================ */

void free_stream(ParticleSystem *sys) {
    for (int i = 0; i < NPART; i++) {
        double E = particle_energy(&sys->part[i]);
        sys->part[i].r = vec3_add(
            sys->part[i].r,
            vec3_scale(sys->part[i].p, DT / E)
        );

        Vec3 wrap = vec3_floor(vec3_scale(sys->part[i].r, 1.0/sys->L));
        sys->part[i].r = vec3_add(sys->part[i].r, vec3_scale(wrap, -sys->L));
    }
}

/* ============================================================
   Collision kernel
   ============================================================ */

void scatter_isotropic(Particle *a, Particle *b) {
    Vec3 P = vec3_add(a->p, b->p);
    double EA = particle_energy(a);
    double EB = particle_energy(b);

    Vec3 beta = vec3_scale(P, 1.0 / (EA + EB));
    Vec3 pA_com = lorentz_boost(a->p, EA, beta);

    double p_mag = vec3_norm(pA_com);
    pA_com = vec3_scale(rng_unit_sphere(), p_mag);

    a->p = lorentz_boost(pA_com, sqrt(p_mag*p_mag + a->m*a->m), vec3_scale(beta, -1));
    b->p = lorentz_boost(vec3_scale(pA_com, -1),
                         sqrt(p_mag*p_mag + b->m*b->m),
                         vec3_scale(beta, -1));
}

int collide_monte_carlo(ParticleSystem *sys) {
    int collisions = 0;
    double prefactor = FM2 * SIGMA_FM2 * DT / sys->V;

    for (int a = 0; a < NPART; a++) {
        int b = a;
        do { b = ((int)(rng_uniform()*NPART))%NPART; } while (b == a);

        double Ea = particle_energy(&sys->part[a]);
        double Eb = particle_energy(&sys->part[b]);
        double ma2 = sys->part[a].m * sys->part[a].m;
        double mb2 = sys->part[b].m * sys->part[b].m;

        double s_ab = Ea*Eb - vec3_dot(sys->part[a].p, sys->part[b].p);

        double v_rel =
            sqrt(fmax(s_ab*s_ab - ma2*mb2, 0.0)) / (Ea*Eb);

        double prob = prefactor * v_rel;

        if (prob * (NPART - 1) / 2.0 > rng_uniform()) {
            scatter_isotropic(&sys->part[a], &sys->part[b]);
            collisions++;
        }
    }

    return collisions;
}

/* ============================================================
   Observables
   ============================================================ */

double bulk_viscous_pressure(const ParticleSystem *sys) {
    double sum = 0.0;
    for (int i = 0; i < NPART; i++) {
        double E = particle_energy(&sys->part[i]);
        sum += \
                sys->part[i].p.x*sys->part[i].p.x / fmax(E, EPS) \
            +   sys->part[i].p.y*sys->part[i].p.y / fmax(E, EPS) \
            +   sys->part[i].p.z*sys->part[i].p.z / fmax(E, EPS);
    }
    return sum / (3.0*sys->V);
}

double shear_stress_tensor_xy(const ParticleSystem *sys) {
    double sum = 0.0;
    for (int i = 0; i < NPART; i++) {
        double E = particle_energy(&sys->part[i]);
        sum += sys->part[i].p.x * sys->part[i].p.y / fmax(E, EPS);
    }
    return sum / sys->V;
}

/* ============================================================
   Autocorrelation Function
   ============================================================ */

double correlator_integral(double* observable) {
    double correlator[TMAX];
    double mean = 0.0;
    for (int i = 0; i < NT; i++) mean += observable[i];
    mean /= NT;

    FILE *f = fopen("output.csv", "w");
    fprintf(f, "time,observable,correlator\n");

    double integral = 0.0;
    for (int t = 0; t < TMAX; t++) {
        correlator[t] = 0.0;
        for (int i = 0; i < NT - t; i++)
            correlator[t] += (observable[i]-mean)*(observable[i+t]-mean);
        correlator[t] /= (NT - t);
        integral += correlator[t];

        fprintf(f, "%f,%e,%e\n", DT*t, observable[t], correlator[t]);
    }
    fclose(f);
    return integral;
}

/* ============================================================
   Main
   ============================================================ */

int main(void) {
    srand(SEED);

    ParticleSystem sys;
    sys.part = malloc(sizeof(Particle) * NPART);
    initialize_particles(&sys);

    double observable[NT];

    double t0 = omp_get_wtime();
    int coll_count = 0;
    for (int t = 0; t < NT; t++) {
        // free_stream(&sys);
        coll_count += collide_monte_carlo(&sys);

        // For any other observable change the definition here
        observable[t] = shear_stress_tensor_xy(&sys);
    }
    double integral = correlator_integral(observable);

    printf("Eta = %f\n", sys.V / TEMPERATURE * integral * DT);
    printf("Runtime = %f s\n", omp_get_wtime() - t0);
    printf("Collision rate = %f\n", coll_count/(double)NT);

    free(sys.part);
    return 0;
}
