#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include "bessel.h"


/* ============================================================
   Constants
   ============================================================ */

int     N1;
int     N2;
double  MASS1;
double  MASS2;
double  TEMPERATURE;
long    SEED;
int     REPEAT;
int     NPART = 1000;
double  DT = 0.001;

__constant__ double  SIGMA11;
__constant__ double  SIGMA22;
__constant__ double  SIGMA12;

#define     PI              3.141592653589793
#define     FM2             25.679
#define     NT              30000
#define     TMAX            500
#define     EPS             1e-20
#define     TEST_RUN        200

/* ============================================================
   Derived thermodynamics
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
    // ****** MIXTURE *******
    int specie;
    // **********************
    Vec3 r;
    Vec3 p;
    double m;
} Particle;

typedef struct {
    Particle *part;
    int NPART;
    double V;
    double L;
    int coll_count;
} ParticleSystem;

/* ============================================================
   Relativistic kinematics
   ============================================================ */

__host__ __device__
static inline double particle_energy(const Particle *part) {
    return sqrt(vec3_dot(part->p, part->p) + part->m * part->m);
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

        // ****** MIXTURE *******
    double n1 =     number_density(1);
    double n2 =     number_density(2);
    double n =      n1+n2;
        // **********************

    sys->V = NPART / n;
    sys->L = cbrt(sys->V);
    sys->NPART = NPART;

    Vec3 p_total = {0,0,0};
    double e_total = 0.0;

    for (int i = 0; i < NPART; i++) {

        // ****** MIXTURE *******
        double MASS;
        if (i < N1)         {MASS = MASS1; sys->part[i].specie = 1;}
        else if (i < N1+N2) {MASS = MASS2; sys->part[i].specie = 2;}
        else                printf("Bad index\n");
        // **********************

        sys->part[i].m = MASS;
        sys->part[i].r = vec3_scale(
            Vec3{rng_uniform(), rng_uniform(), rng_uniform()},
            sys->L
        );

        double p, e;
        do {
            do {
                p = -TEMPERATURE * log(rng_uniform()*rng_uniform()*rng_uniform());
            } while (isnan(p) || isinf(p));

            e = sqrt(p*p + MASS*MASS); 

        } while (rng_uniform() >= exp(-(e - p)/TEMPERATURE));

        sys->part[i].p = vec3_scale(rng_unit_sphere(), p);

        p_total = vec3_add(p_total, sys->part[i].p);
        e_total += e;
    }
    // boost to Landau rest frame
    Vec3 beta = vec3_scale(p_total, 1.0 / fmax(e_total, EPS));
    // ****** ROBUST *******
    double pSQR1=0.0, pSQR2=0.0;
    // **********************
    for (int i = 0; i < NPART; i++) {
        double E = particle_energy(&sys->part[i]);
        sys->part[i].p = lorentz_boost(sys->part[i].p, E, beta);
        
        // ****** ROBUST *******
        if (i < N1)         {pSQR1 += vec3_dot(sys->part[i].p, sys->part[i].p);}
        else if (i < N1+N2) {pSQR2 += vec3_dot(sys->part[i].p, sys->part[i].p);}
        else                printf("Bad index\n");
        // **********************
    }
    // ****** ROBUST *******
    pSQR1 /= N1;
    pSQR2 /= N2;
    double z, K2, K3, mean_pSQR;
    // rescale momentum to match expected value at temperature
    // ****** SPECIE 1 ******
    if (MASS1 == 0) {
      mean_pSQR = 12.0 * TEMPERATURE * TEMPERATURE;
    }
    else {
      z = MASS1 / TEMPERATURE;
      K2 = bessel_Kn(2, z);
      K3 = bessel_Kn(3, z);
      mean_pSQR = 3.0 * MASS1 * TEMPERATURE * (K3 / K2);
    }
    // printf("momentum scaling factor: %f\n", sqrt(mean_pSQR/pSQR1));
    for (int i = 0; i < N1; i++) {
        sys->part[i].p = vec3_scale(sys->part[i].p, sqrt(mean_pSQR/pSQR1));
    }
    // rescale momentum to match expected value at temperature
    // ****** SPECIE 2 ******
    if (MASS2 == 0) {
      mean_pSQR = 12.0 * TEMPERATURE * TEMPERATURE;
    }
    else {
      z = MASS2 / TEMPERATURE;
      K2 = bessel_Kn(2, z);
      K3 = bessel_Kn(3, z);
      mean_pSQR = 3.0 * MASS2 * TEMPERATURE * (K3 / K2);
    }
    // printf("momentum scaling factor: %f\n", sqrt(mean_pSQR/pSQR2));
    for (int i = N1; i < N1+N2; i++) {
        sys->part[i].p = vec3_scale(sys->part[i].p, sqrt(mean_pSQR/pSQR2));
    }
    // **********************
}

/* ============================================================
   Free streaming
   ============================================================ */

__global__ void free_stream(ParticleSystem *sys, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sys->NPART) { 
        double E = particle_energy(&sys->part[idx]);
        sys->part[idx].r = vec3_add(
            sys->part[idx].r,
            vec3_scale(sys->part[idx].p, dt / fmax(E,EPS))
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

__global__ void collide_monte_carlo(ParticleSystem *sys, double dt, curandStatePhilox4_32_10_t *states, int seed, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sys->NPART) { 
        // ****** MIXTURE *******
        double prefactor = FM2 * dt / sys->V;
        // **********************

        int a = idx;
        int b = (a*(sys->NPART-1) + seed)% sys->NPART;
        if (a < b) {
            double Ea = particle_energy(&sys->part[a]);
            double Eb = particle_energy(&sys->part[b]);
            double ma2 = sys->part[a].m * sys->part[a].m;
            double mb2 = sys->part[b].m * sys->part[b].m;

            double s_ab = Ea*Eb - vec3_dot(sys->part[a].p, sys->part[b].p);

            double v_rel =
                sqrt(fmax(s_ab*s_ab - ma2*mb2, 0.0)) / fmax(Ea*Eb, EPS);

            // ****** MIXTURE *******
            double prob = prefactor * sigma(sys->part[a].specie, sys->part[b].specie) * v_rel;
            // **********************

            if (prob * sys->NPART / (double)repeat > curand_uniform_double(&states[idx])) {
                scatter_isotropic(&sys->part[a], &sys->part[b], &states[idx]);
                atomicAdd(&sys->coll_count, 1);
            }
        }
    }
}

/* ============================================================
   Observables
   ============================================================ */

__global__ void get_observable(const ParticleSystem *sys, double *observable, int selector, int t) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int threadNum = blockDim.x * gridDim.x;

    double sum = 0.;
    for (int i=tid; i<sys->NPART; i+=threadNum) {
        double E = particle_energy(&sys->part[i]);
        double px = sys->part[i].p.x;
        double py = sys->part[i].p.y;
        double pz = sys->part[i].p.z;

        // Shear Stress Tensor xy
        if          (selector == 0)     sum += px*py / fmax(E, EPS);
        
        // Shear Stress Tensor yz
        else if     (selector == 1)     sum += py*pz / fmax(E, EPS);
        
        // Shear Stress Tensor zx
        else if     (selector == 2)     sum += pz*px / fmax(E, EPS);

        // Bulk Viscous Pressure
        else if     (selector == 3)     sum += (px*px + py*py + pz*pz) / (3.0*fmax(E, EPS));
    }

    extern __shared__ double s[];
    s[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int sstep = blockDim.x/2; sstep>0; sstep>>=1) {
        if (threadIdx.x < sstep) {
            s[threadIdx.x] += s[threadIdx.x + sstep];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&observable[t], s[0]/sys->V);
}

/* ============================================================
   Autocorrelation Function
   ============================================================ */

double correlator_integral(double* observable, char* csv_label=NULL) {
    double correlator[TMAX];
    double mean = 0.0;
    for (int i = 0; i < NT; i++) mean += observable[i];
    mean = mean / (double)NT;

    FILE *fp = NULL;
    if (csv_label != NULL) {
        char path[256];
        snprintf(path, sizeof(path), "./results/%s", csv_label);
        fp = fopen(path, "w");
        if (fp) fprintf(fp, "time,tensor,correlator\n");
    }

    double integral = 0.0;
    for (int t = 0; t < TMAX; t++) {
        correlator[t] = 0.0;
        for (int i = 0; i < NT - t; i++)
            correlator[t] += (observable[i]-mean)*(observable[i+t]-mean);
        correlator[t] /= (double)(NT - t);
        integral += correlator[t];

        if (fp) fprintf(fp, "%f,%e,%e\n", DT*t, observable[t], correlator[t]);
    }
    if (fp) fclose(fp);
    return integral - 0.5*correlator[0];
}

/* ============================================================
   RNG Initialize
   ============================================================ */

__global__ void init_rng(curandStatePhilox4_32_10_t *states, int NPART, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NPART) curand_init(seed, idx, 0, &states[idx]);
}

/* ============================================================
   Main
   ============================================================ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s input_file\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];  // input filename from command line
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open input file");
        return 1;
    }
    
    char name[100], ext[100];
    sscanf(filename, "./results/%[^.].%s", name, ext);

    double x1, s11, s22, s12;
    char key[50];

    fscanf(fp, "%s %d", key, &SEED);
    fscanf(fp, "%s %lf", key, &x1);
    fscanf(fp, "%s %lf", key, &TEMPERATURE);
    fscanf(fp, "%s %lf", key, &s11);
    fscanf(fp, "%s %lf", key, &s22);
    fscanf(fp, "%s %lf", key, &s12);
    fscanf(fp, "%s %lf", key, &MASS1);
    fscanf(fp, "%s %lf", key, &MASS2);
    fscanf(fp, "%s %d", key, &REPEAT);

    fclose(fp);

    N1 = NPART * x1;
    N2 = NPART * (1.0-x1);
    cudaMemcpyToSymbol(SIGMA11, &s11, sizeof(double));
    cudaMemcpyToSymbol(SIGMA22, &s22, sizeof(double));
    cudaMemcpyToSymbol(SIGMA12, &s12, sizeof(double));

    srand(SEED);
    double observable[NT];
    
    ParticleSystem sys_host;
    Particle* part_host = (Particle*) malloc(sizeof(Particle) * NPART);
    sys_host.part = part_host;

    ParticleSystem* sys_dev;
    Particle* part_dev;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    initialize_particles(&sys_host);
    cudaMalloc(&sys_dev,    sizeof(ParticleSystem));
    cudaMalloc(&part_dev,   sizeof(Particle) * NPART);

    cudaMemcpy(part_dev, part_host, sizeof(Particle) * NPART, cudaMemcpyHostToDevice);
    sys_host.part = part_dev;
    sys_host.coll_count = 0;
    cudaMemcpy(sys_dev, &sys_host, sizeof(ParticleSystem), cudaMemcpyHostToDevice);
    // sys_host.part = part_host;

    
    double *shear_xy, *shear_yz, *shear_zx, *bulk;
    cudaMalloc(&shear_xy, sizeof(double) * NT);
    cudaMalloc(&shear_yz, sizeof(double) * NT);
    cudaMalloc(&shear_zx, sizeof(double) * NT);
    cudaMalloc(&bulk,     sizeof(double) * NT);

    cudaMemset(shear_xy, 0, sizeof(double) * NT);
    cudaMemset(shear_yz, 0, sizeof(double) * NT);
    cudaMemset(shear_zx, 0, sizeof(double) * NT);
    cudaMemset(bulk,     0, sizeof(double) * NT);

    curandStatePhilox4_32_10_t *d_states;
    cudaMalloc(&d_states, NPART * sizeof(curandStatePhilox4_32_10_t));
    init_rng<<<(NPART+255)/256, 256>>>(d_states, NPART, SEED);

    for (int t = 0; t < NT; t++) {
        
        // free_stream<<<(NPART+255)/256, 256>>>(sys_dev, DT);

        for (int r = 0; r < REPEAT; r++) {
            collide_monte_carlo<<<(NPART+255)/256, 256>>>(sys_dev, DT, d_states, rand(), REPEAT);
        }

        if (t == TEST_RUN && DT < 1e7) {
            cudaMemcpy(
                &sys_host, 
                sys_dev, 
                sizeof(ParticleSystem), 
                cudaMemcpyDeviceToHost
            );

            if (sys_host.coll_count/(double)NPART < 2) {
                t = 0;
                DT *= 2.0;
                sys_host.coll_count = 0;
                cudaMemcpy(
                    sys_dev, 
                    &sys_host, 
                    sizeof(ParticleSystem), 
                    cudaMemcpyHostToDevice
                );
                
                cudaMemset(shear_xy, 0, sizeof(double) * TEST_RUN);
                cudaMemset(shear_yz, 0, sizeof(double) * TEST_RUN);
                cudaMemset(shear_zx, 0, sizeof(double) * TEST_RUN);
                cudaMemset(bulk,     0, sizeof(double) * TEST_RUN);
            }
        }

        // For any other observable change the third argument here: 0,1,2 for shear stress tensor, 3 for bulk viscous pressure
        get_observable<<<(NPART+255)/256, 256, 256 * sizeof(double)>>>(sys_dev, shear_xy, 0, t);
        get_observable<<<(NPART+255)/256, 256, 256 * sizeof(double)>>>(sys_dev, shear_yz, 1, t);
        get_observable<<<(NPART+255)/256, 256, 256 * sizeof(double)>>>(sys_dev, shear_zx, 2, t);
        get_observable<<<(NPART+255)/256, 256, 256 * sizeof(double)>>>(sys_dev, bulk, 3, t);
    }
    
    cudaMemcpy(&sys_host, sys_dev, sizeof(ParticleSystem), cudaMemcpyDeviceToHost); // get sys_host.coll_count

    char out[256] = "bulk_";
    strcat(out, name);
    strcat(out, ".csv");
    cudaMemcpy(observable, bulk, sizeof(double) * NT, cudaMemcpyDeviceToHost);
    double zeta = sys_host.V/TEMPERATURE * correlator_integral(observable, out) * DT;

    strcpy(out, "shear_");
    strcat(out, name);
    strcat(out, ".csv");
    cudaMemcpy(observable, shear_xy, sizeof(double) * NT, cudaMemcpyDeviceToHost);
    double integral_xy = correlator_integral(observable,  out);
    
    cudaMemcpy(observable, shear_yz, sizeof(double) * NT, cudaMemcpyDeviceToHost);
    double integral_yz = correlator_integral(observable);

    cudaMemcpy(observable, shear_zx, sizeof(double) * NT, cudaMemcpyDeviceToHost);
    double integral_zx = correlator_integral(observable);

    double eta = sys_host.V/TEMPERATURE * (integral_xy+integral_yz+integral_zx)/3.0 * DT;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Eta: %e, Zeta: %e\n", eta, zeta);
    printf("Dt: %f, Kernel time: %.3f ms\n", DT, ms);
    printf("Coll rate: %f\n", sys_host.coll_count/(double)NT);

    cudaFree(shear_xy);
    cudaFree(shear_yz);
    cudaFree(shear_zx);
    cudaFree(bulk);

    cudaFree(part_dev);
    cudaFree(sys_dev);
    free(part_host);
    return 0;
}
