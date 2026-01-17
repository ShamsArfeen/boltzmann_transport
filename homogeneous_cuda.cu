#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>


/* ============================================================
   Constants
   ============================================================ */

#define     SEED            4432336
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

__host__
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
    return integral*dt*pow(MASS,4) / (double)(TEMPERATURE* 2*PI * PI);
}

/* ============================================================
   Vector algebra
   ============================================================ */

struct Vec3 {
    double x, y, z;
};

__host__ __device__
static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return Vec3{a.x+b.x, a.y+b.y, a.z+b.z};
}

__host__ __device__
static inline Vec3 vec3_scale(Vec3 a, double s) {
    return Vec3{a.x*s, a.y*s, a.z*s};
}

__host__ __device__
static inline double vec3_dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__
static inline double vec3_norm(Vec3 a) {
    return sqrt(vec3_dot(a,a));
}

__host__ __device__
static inline Vec3 vec3_floor(Vec3 a) {
    return Vec3{floor(a.x), floor(a.y), floor(a.z)};
}

/* ============================================================
   RNG utilities
   ============================================================ */

static inline double rng_uniform(void) {
    return rand() / (double)RAND_MAX;
}

__host__
static inline double rng_normal(void) {
    double u, v, s;
    do {
        u = 2.0 * rng_uniform() - 1.0;
        v = 2.0 * rng_uniform() - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s < EPS);
    return u * sqrt(-2.0 * log(s) / s);
}

__host__
static inline Vec3 rng_unit_sphere(void) {
    Vec3 v = {rng_normal(), rng_normal(), rng_normal()};
    return vec3_scale(v, 1.0 / vec3_norm(v));
}

__device__
static inline Vec3 curng_unit_sphere(curandStatePhilox4_32_10_t *state) {
    Vec3 v = {curand_normal_double(state), curand_normal_double(state), curand_normal_double(state)};
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
    int coll_count;
} ParticleSystem;

/* ============================================================
   Relativistic kinematics
   ============================================================ */

__host__ __device__
static inline double particle_energy(const Particle *part) {
    return sqrt(fmax(vec3_dot(part->p, part->p) + part->m * part->m, 0.0));
}

__host__ __device__
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
            Vec3{rng_uniform(), rng_uniform(), rng_uniform()},
            sys->L
        );

        double beta_e;
        double beta_m = MASS / (double)TEMPERATURE;
        while (1) {
            beta_e = -log(rng_uniform()*rng_uniform()*rng_uniform());
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

__global__ void free_stream(ParticleSystem *sys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NPART) { 
        double E = particle_energy(&sys->part[idx]);
        sys->part[idx].r = vec3_add(
            sys->part[idx].r,
            vec3_scale(sys->part[idx].p, DT / E)
        );

        Vec3 wrap = vec3_floor(vec3_scale(sys->part[idx].r, 1.0/sys->L));
        sys->part[idx].r = vec3_add(sys->part[idx].r, vec3_scale(wrap, -sys->L));
    }
}

/* ============================================================
   Collision kernel
   ============================================================ */

__device__
static inline void scatter_isotropic(Particle *a, Particle *b, curandStatePhilox4_32_10_t *state) {
    Vec3 P = vec3_add(a->p, b->p);
    double EA = particle_energy(a);
    double EB = particle_energy(b);

    Vec3 beta = vec3_scale(P, 1.0 / fmax(EA + EB, EPS));
    Vec3 pA_com = lorentz_boost(a->p, EA, beta);

    double p_mag = vec3_norm(pA_com);
    pA_com = vec3_scale(curng_unit_sphere(state), p_mag);

    a->p = lorentz_boost(pA_com, sqrt(p_mag*p_mag + a->m*a->m), vec3_scale(beta, -1));
    b->p = lorentz_boost(vec3_scale(pA_com, -1),
                         sqrt(fmax(p_mag*p_mag + b->m*b->m, 0.0)),
                         vec3_scale(beta, -1));
}

__global__ void collide_monte_carlo(ParticleSystem *sys, curandStatePhilox4_32_10_t *states, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NPART) { 
        double prefactor = FM2 * SIGMA_FM2 * DT / sys->V;

        int a = idx;
        int b = (a*(NPART-1) + k)%NPART;
        if (a < b) {
            double Ea = particle_energy(&sys->part[a]);
            double Eb = particle_energy(&sys->part[b]);
            double ma2 = sys->part[a].m * sys->part[a].m;
            double mb2 = sys->part[b].m * sys->part[b].m;

            double s_ab = Ea*Eb - vec3_dot(sys->part[a].p, sys->part[b].p);

            double v_rel =
                sqrt(fmax(s_ab*s_ab - ma2*mb2, 0.0)) / fmax(Ea*Eb, EPS);

            double prob = prefactor * v_rel;

            if (prob * NPART > curand_uniform_double(&states[idx])) {
                scatter_isotropic(&sys->part[a], &sys->part[b], &states[idx]);
                atomicAdd(&sys->coll_count, 1);
            }
        }
    }
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
    fprintf(f, "time,tensor,correlator\n");

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
   RNG Initialize
   ============================================================ */

__global__ void init_rng(curandStatePhilox4_32_10_t *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NPART) curand_init(seed, idx, 0, &states[idx]);
}

/* ============================================================
   Main
   ============================================================ */

int main(void) {
    srand(SEED);
    double observable[NT];
    
    ParticleSystem sys_host;
    Particle* part_host = (Particle*) malloc(sizeof(Particle) * NPART);
    sys_host.part = part_host;

    ParticleSystem* sys_dev;
    Particle* part_dev;
    
    double t0 = omp_get_wtime();

    initialize_particles(&sys_host);
    cudaMalloc(&sys_dev, sizeof(ParticleSystem));
    cudaMalloc(&part_dev, sizeof(Particle) * NPART);

    cudaMemcpy(part_dev, part_host, sizeof(Particle) * NPART, cudaMemcpyHostToDevice);
    sys_host.part = part_dev;
    sys_host.coll_count = 0;
    cudaMemcpy(sys_dev, &sys_host, sizeof(ParticleSystem), cudaMemcpyHostToDevice);
    sys_host.part = part_host;

    curandStatePhilox4_32_10_t *d_states;
    cudaMalloc(&d_states, NPART * sizeof(curandStatePhilox4_32_10_t));
    init_rng<<<(NPART+255)/256, 256>>>(d_states, SEED);

    for (int t = 0; t < NT; t++) {

        free_stream<<<(NPART+255)/256, 256>>>(sys_dev);
        int k = rand();
        collide_monte_carlo<<<(NPART+255)/256, 256>>>(sys_dev, d_states, k);
        cudaDeviceSynchronize();
        cudaMemcpy(part_host, part_dev, sizeof(Particle) * NPART, cudaMemcpyDeviceToHost);

        // For any other observable change the definition here
        observable[t] = shear_stress_tensor_xy(&sys_host);
    }
    cudaMemcpy(&sys_host, sys_dev, sizeof(ParticleSystem), cudaMemcpyDeviceToHost);
    double integral = correlator_integral(observable);

    printf("Eta = %f\n", sys_host.V / TEMPERATURE * integral * DT);
    printf("Runtime = %f s\n", omp_get_wtime() - t0);
    printf("Collision rate = %f\n", sys_host.coll_count/(double)NT);

    cudaFree(part_dev);
    cudaFree(sys_dev);
    free(part_host);
    return 0;
}
