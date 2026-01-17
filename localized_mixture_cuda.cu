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

        // ****** MIXTURE *******
#define     N1              1000
#define     N2              1000
#define     MASS1           0.0
#define     MASS2           0.0
#define     SIGMA11         0.3
#define     SIGMA22         0.6
#define     SIGMA12         0.9
        // **********************

#define SEED        3322555
#define PI          3.141592653589793
#define TEMPERATURE 0.6
#define FM2         25.679

#define NPART   ((N1)+(N2))
#define NT      50000
#define NROW    3
#define TMAX    500
#define DT      0.1

#define EPS     1e-12
#define NCELL   (NROW*NROW*NROW)
#define CELLCAP (4*(NPART)/(NCELL)+10)

/* ============================================================
   Thermodynamics
   ============================================================ */

__host__
static inline double number_density(int specie) {
    double MASS;
    // ****** MIXTURE *******
    if (specie == 1) MASS = MASS1;
    else if (specie == 2) MASS = MASS2;
    else printf("Invalid specie\n");
    // **********************

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

__device__ 
static inline double sigma(int specie_a, int specie_b) {
    // ****** MIXTURE *******
    if (specie_a == 1 && specie_b == 1)             return SIGMA11;
    else if (specie_a == 2 && specie_b == 2)        return SIGMA22;
    else                                            return SIGMA12;
    // **********************
}

/* ============================================================
   Vector algebra
   ============================================================ */

struct Vec3 { double x, y, z; };

__host__ __device__ static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return {a.x+b.x, a.y+b.y, a.z+b.z};
}

__host__ __device__ static inline Vec3 vec3_scale(Vec3 a, double s) {
    return {a.x*s, a.y*s, a.z*s};
}

__host__ __device__ static inline double vec3_dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ static inline double vec3_norm(Vec3 a) {
    return sqrt(vec3_dot(a, a));
}

__host__ __device__ static inline Vec3 vec3_floor(Vec3 a) {
    return {floor(a.x), floor(a.y), floor(a.z)};
}

/* ============================================================
   RNG
   ============================================================ */

static inline double rng_uniform(void) {
    return rand() / (double)RAND_MAX;
}

__host__ static inline double rng_normal(void) {
    double u, v, s;
    do {
        u = 2.0*rng_uniform() - 1.0;
        v = 2.0*rng_uniform() - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s < EPS);
    return u * sqrt(-2.0 * log(s) / s);
}

__host__ static inline Vec3 rng_unit_sphere(void) {
    Vec3 v = {rng_normal(), rng_normal(), rng_normal()};
    return vec3_scale(v, 1.0 / vec3_norm(v));
}

__device__ static inline Vec3 curng_unit_sphere(curandStatePhilox4_32_10_t *st) {
    Vec3 v = {
        curand_normal_double(st),
        curand_normal_double(st),
        curand_normal_double(st)
    };
    return vec3_scale(v, 1.0 / vec3_norm(v));
}

/* ============================================================
   Particle structures
   ============================================================ */

typedef struct {
    // ****** MIXTURE *******
    int specie;
    // **********************
    Vec3 r, p;
    double m;
    int cell_id;     // spatial cell index
    int cell_slot;   // index inside cell list
} Particle;

typedef struct {
    Particle *part;
    double V, DV, L;
    int collision_count;
    int cell_occupancy[NCELL];
    int *cell_particles;
} ParticleSystem;

/* ============================================================
   Relativistic kinematics
   ============================================================ */

__host__ __device__ static inline double particle_energy(const Particle *p) {
    return sqrt(fmax(vec3_dot(p->p, p->p) + p->m*p->m, 0.0));
}

__host__ __device__ static inline Vec3 lorentz_boost(Vec3 p, double E, Vec3 beta) {
    double b2 = vec3_dot(beta, beta);
    if (b2 < EPS) return p;

    double gamma = 1.0 / sqrt(1.0 - b2);
    double pb = vec3_dot(p, beta);

    return vec3_add(p, vec3_scale(beta, (gamma-1.0)*pb/b2 - gamma*E));
}

/* ============================================================
   Initialization
   ============================================================ */

void initialize_particles(ParticleSystem *sys) {

        // ****** MIXTURE *******
    double n1 =     number_density(1);
    double n2 =     number_density(2);
    double n =      n1+n2;
        // **********************

    sys->V = NPART / n;
    sys->DV = sys->V / NCELL;
    sys->L = cbrt(sys->V);

    Vec3 Ptot = {0,0,0};
    double Etot = 0.0;

    for (int i = 0; i < NPART; i++) {

        // ****** MIXTURE *******
        double MASS;
        if (i < N1)         {MASS = MASS1; sys->part[i].specie = 1;}
        else if (i < N1+N2) {MASS = MASS2; sys->part[i].specie = 2;}
        else                printf("Bad index\n");
        // **********************

        Particle *p = &sys->part[i];
        p->m = MASS;
        p->r = vec3_scale({rng_uniform(), rng_uniform(), rng_uniform()}, sys->L);

        double beta_e, beta_m = MASS / TEMPERATURE;
        do {
            beta_e = -log(rng_uniform()*rng_uniform()*rng_uniform() + EPS);
        } while (beta_e < beta_m ||
                 rng_uniform() > sqrt(fmax(beta_e*beta_e - beta_m*beta_m, 0.0)) / beta_e);

        double E = beta_e * TEMPERATURE;
        double p_mag = sqrt(E*E - MASS*MASS);
        p->p = vec3_scale(rng_unit_sphere(), p_mag);

        Ptot = vec3_add(Ptot, p->p);
        Etot += E;
    }

    Vec3 beta = vec3_scale(Ptot, 1.0 / fmax(Etot, EPS));
    for (int i = 0; i < NPART; i++) {
        Particle *p = &sys->part[i];
        p->p = lorentz_boost(p->p, particle_energy(p), beta);
    }
}

/* ============================================================
   Cell binning
   ============================================================ */

__global__ void bin_particles_into_cells(ParticleSystem *sys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NPART) return;

    Vec3 cell = vec3_floor(vec3_scale(sys->part[i].r, 0.99 * NROW / sys->L));
    int cell_id = cell.x*NROW*NROW + cell.y*NROW + cell.z;

    int slot = atomicAdd(&sys->cell_occupancy[cell_id], 1);
    sys->cell_particles[cell_id*CELLCAP + slot] = i;

    sys->part[i].cell_id   = cell_id;
    sys->part[i].cell_slot = slot;
}

/* ============================================================
   Free streaming
   ============================================================ */

__global__ void free_stream(ParticleSystem *sys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NPART) return;

    Particle *p = &sys->part[i];
    double invE = DT / particle_energy(p);

    p->r = vec3_add(p->r, vec3_scale(p->p, invE));

    Vec3 wrap = vec3_floor(vec3_scale(p->r, 1.0 / sys->L));
    p->r = vec3_add(p->r, vec3_scale(wrap, -sys->L));
}

/* ============================================================
   Collisions
   ============================================================ */

__device__ static inline void scatter_isotropic(
    Particle *a, Particle *b, curandStatePhilox4_32_10_t *st)
{
    Vec3 P = vec3_add(a->p, b->p);
    double Ea = particle_energy(a);
    double Eb = particle_energy(b);

    Vec3 beta = vec3_scale(P, 1.0 / fmax(Ea + Eb, EPS));
    Vec3 p_com = lorentz_boost(a->p, Ea, beta);

    double p_mag = vec3_norm(p_com);
    p_com = vec3_scale(curng_unit_sphere(st), p_mag);

    a->p = lorentz_boost(p_com, sqrt(p_mag*p_mag + a->m*a->m), vec3_scale(beta, -1));
    b->p = lorentz_boost(vec3_scale(p_com, -1),
                         sqrt(p_mag*p_mag + b->m*b->m),
                         vec3_scale(beta, -1));
}

__global__ void collide_monte_carlo(
    ParticleSystem *sys,
    curandStatePhilox4_32_10_t *states,
    int rng_shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NPART) return;

    Particle *pi = &sys->part[i];
    int cell_id = pi->cell_id;
    int slot_a  = pi->cell_slot;
    int n = sys->cell_occupancy[cell_id];

    if (n <= 1) return;

    int *cell_list = &sys->cell_particles[cell_id * CELLCAP];
    int slot_b = ((n - 1)*slot_a + (rng_shift % n)) % n;

    if (slot_a <= slot_b) return;

    int a = cell_list[slot_a];
    int b = cell_list[slot_b];

    Particle *pa = &sys->part[a];
    Particle *pb = &sys->part[b];

    double Ea = particle_energy(pa);
    double Eb = particle_energy(pb);

    double s = Ea*Eb - vec3_dot(pa->p, pb->p);
    double vrel = sqrt(fmax(s*s - pa->m*pa->m * pb->m*pb->m, 0.0))
                  / fmax(Ea*Eb, EPS);

    // ****** MIXTURE *******
    double SIGMA = sigma(sys->part[a].specie, sys->part[b].specie);
    double prob = FM2 * SIGMA * DT * vrel / sys->DV;
    // **********************

    if (prob * n > curand_uniform_double(&states[i])) {
        scatter_isotropic(pa, pb, &states[i]);
        atomicAdd(&sys->collision_count, 1);
    }
}

/* ============================================================
   Observables
   ============================================================ */

double bulk_pressure(const ParticleSystem *sys) {
    double trace_p2_over_E = 0.0;

    for (int i = 0; i < NPART; i++) {
        const Particle *p = &sys->part[i];
        double invE = 1.0 / fmax(particle_energy(p), EPS);

        trace_p2_over_E +=
            (p->p.x*p->p.x + p->p.y*p->p.y + p->p.z*p->p.z) * invE;
    }
    return trace_p2_over_E / (3.0 * sys->V);
}

double shear_xy(const ParticleSystem *sys) {
    double sum_pxp_y_over_E = 0.0;

    for (int i = 0; i < NPART; i++) {
        const Particle *p = &sys->part[i];
        sum_pxp_y_over_E +=
            p->p.x * p->p.y / fmax(particle_energy(p), EPS);
    }
    return sum_pxp_y_over_E / sys->V;
}

/* ============================================================
   Autocorrelation / Green–Kubo integral
   ============================================================ */

double time_correlator_integral(const double *signal) {
    double mean = 0.0;
    for (int i = 0; i < NT; i++)
        mean += signal[i];
    mean /= NT;

    FILE *out = fopen("output.csv", "w");
    fprintf(out, "time,observable,correlator\n");

    double integral = 0.0;

    for (int t = 0; t < TMAX; t++) {
        double C_t = 0.0;
        int n = NT - t;

        for (int i = 0; i < n; i++) {
            double d0 = signal[i]     - mean;
            double dt = signal[i + t] - mean;
            C_t += d0 * dt;
        }

        C_t /= n;
        integral += C_t;

        fprintf(out, "%f,%e,%e\n", DT * t, signal[t], C_t);
    }

    fclose(out);
    return integral;
}

/* ============================================================
   RNG init
   ============================================================ */

__global__ void init_rng(curandStatePhilox4_32_10_t *states, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NPART)
        curand_init(seed, i, 0, &states[i]);
}


/* ============================================================
   Main
   ============================================================ */

int main(void) {
    srand(SEED);

    double observable[NT];
    double t0 = omp_get_wtime();

    /* ---------------- Host initialization ---------------- */

    ParticleSystem sys_host = {0};
    Particle* part_host = (Particle*)malloc(sizeof(Particle) * NPART);
    sys_host.part = part_host;
    initialize_particles(&sys_host);

    /* ---------------- Device allocation ---------------- */

    ParticleSystem *sys_dev;
    Particle *part_dev;
    int *cell_particles_dev;
    curandStatePhilox4_32_10_t *rng_states;

    cudaMalloc(&sys_dev, sizeof(ParticleSystem));
    cudaMalloc(&part_dev, sizeof(Particle) * NPART);
    cudaMalloc(&cell_particles_dev, sizeof(int) * CELLCAP * NCELL);
    cudaMalloc(&rng_states, sizeof(curandStatePhilox4_32_10_t) * NPART);

    cudaMemcpy(part_dev, sys_host.part,
               sizeof(Particle) * NPART, cudaMemcpyHostToDevice);

    init_rng<<<(NPART+255)/256, 256>>>(rng_states, SEED);

    /* ---------------- Time evolution ---------------- */

    int total_collisions = 0;

    for (int t = 0; t < NT; t++) {

        sys_host.part              = part_dev;
        sys_host.cell_particles    = cell_particles_dev;
        sys_host.collision_count   = 0;

        for (int c = 0; c < NCELL; c++)
            sys_host.cell_occupancy[c] = 0;

        cudaMemcpy(sys_dev, &sys_host,
                   sizeof(ParticleSystem), cudaMemcpyHostToDevice);

        free_stream<<<(NPART+255)/256, 256>>>(sys_dev);
        bin_particles_into_cells<<<(NPART+255)/256, 256>>>(sys_dev);
        collide_monte_carlo<<<(NPART+255)/256, 256>>>(
            sys_dev, rng_states, rand()
        );

        cudaMemcpy(&sys_host, sys_dev,
                   sizeof(ParticleSystem), cudaMemcpyDeviceToHost);
        sys_host.part = part_host;
        cudaMemcpy(sys_host.part, part_dev,
                   sizeof(Particle) * NPART, cudaMemcpyDeviceToHost);
                   
        total_collisions += sys_host.collision_count;
        observable[t] = shear_xy(&sys_host);
    }

    /* ---------------- Analysis ---------------- */

    double integral = time_correlator_integral(observable);

    printf("Eta = %f\n", sys_host.V / TEMPERATURE * integral * DT);
    printf("Runtime = %f s\n", omp_get_wtime() - t0);
    printf("Collision rate = %f\n", total_collisions / (double)NT);

    /* ---------------- Cleanup ---------------- */

    cudaFree(cell_particles_dev);
    cudaFree(rng_states);
    cudaFree(part_dev);
    cudaFree(sys_dev);
    free(sys_host.part);

    return 0;
}

