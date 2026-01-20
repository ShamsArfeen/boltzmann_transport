# Boltzmann-Transport

An algorithm for evolving Boltzmann systems and extracting transport coefficients using the Green-Kubo method.  
The codes are designed for near-equilibrium relativistic gases and support both homogeneous and spatially
inhomogeneous (or local) systems, with CPU and CUDA implementations.

The primary goal of this repository is to provide minimal, transparent, and extensible reference implementations for
shear and bulk viscosity calculations from microscopic dynamics.

---

## Features

- Green-Kubo evaluation of shear and bulk viscosities
- Relativistic Boltzmann dynamics
- Homogeneous and spatially inhomogeneous (local) systems
- Single-component and multi-component (mixture) gases
- C and CUDA implementations
- Scales to large particle numbers on GPU
- Modular structure with clear extension points

---

## Solver Overview

The repository contains six main solvers, organized by physical model and execution backend.

### 1. Homogeneous single-component gas (CPU)

**File:** `homogeneous.c`

- Single-component relativistic gas
- Homogeneous system (no spatial dependence)
- Collisions occur between any particle pair in the full volume
- Collision probability ∝ Δt / V
- Free streaming does not affect observables
- Computes shear and bulk viscosities via Green-Kubo relations

This solver is intended as a minimal reference implementation.

---

### 2. Homogeneous single-component gas (CUDA)

**File:** `homogeneous_cuda.cu`

- CUDA version of the homogeneous solver
- One GPU thread per particle
- Significantly improved scaling with particle number
- Same physical model as the CPU version

---

### 3. Homogeneous multi-component gas (CPU)

**File:** `mixture.c`

- Multi-component relativistic gas (currently binary mixture)
- Species-dependent masses
- Species-dependent cross sections for each collision channel
- Easily extensible to more components

Regions that require modification for additional species are marked with: MIXTURE

---

### 4. Homogeneous multi-component gas (CUDA)

**File:** `mixture_cuda.cu`

- CUDA implementation of the homogeneous mixture solver
- One thread per particle
- Supports species-dependent masses and cross sections
- Designed for large-scale systems

---

### 5. Spatially inhomogeneous single-component gas (CUDA)

**File:** `localized_cuda.cu`

- Spatial domain divided into cells
- Each cell is treated as locally equilibrated
- Collisions restricted to particle pairs within the same cell
- Designed for non-homogeneous systems
- Uses a cell-based data structure and self-inverse hash functions
  to avoid race conditions during collision handling on GPU

This solver allows the study of local transport properties in non-uniform systems.

---

### 6. Spatially inhomogeneous multi-component gas (CUDA)

**File:** `localized_mixture_cuda.cu`

- Extension of the local solver to multi-component gases
- Species-dependent masses and cross sections
- Collision handling restricted to cells
- Efficient and race-condition-safe GPU implementation

As with the homogeneous mixture solvers, multi-component logic is marked with: MIXTURE

---

## Observables and Output Format

The solvers compute equilibrium correlation functions of components of the energy–momentum tensor, including:

- Shear stress tensor components
- Bulk viscous pressure

Transport coefficients are obtained via time integrals of autocorrelation functions using the Green–Kubo formalism.

---

### Output file

The program outputs a CSV file named `output.csv` with the following format:

| Column | Description                    |
|--------|--------------------------------|
| time   | Simulation time (units of Δt)  |
| observable | Value of the observable at that time |
| correlator | Autocorrelation function value at that time lag |

The file contains exactly `TMAX` rows (not counting the header). The columns have the header: time, observable, correlator

---

## Build Notes

- C codes require a standard C compiler (e.g. `gcc`)
- CUDA codes require `nvcc`
- No external dependencies beyond CUDA and the C standard library

---

## License

This project is released under the MIT License.  
See the `LICENSE` file for details.

---

## Citation

If you use this code in academic work, please cite it or reference the repository.
A `CITATION.cff` file can be added upon request.
