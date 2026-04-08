# Boltzmann-Transport

An algorithm for evolving Boltzmann systems and extracting transport coefficients using the Green-Kubo method.  
The codes are designed for near-equilibrium relativistic gases and support both homogeneous and spatially
inhomogeneous (or local) systems, with CPU and CUDA implementations.

The primary goal of this repository is to provide minimal, transparent, and extensible reference implementations for
shear and bulk viscosity calculations from microscopic dynamics.

---

## Solver Features

- Relativistic Boltzmann dynamics
- Green-Kubo evaluation of shear and bulk viscosities
- Multi-component relativistic gas (currently binary mixture)
- Species-dependent masses
- Species-dependent cross sections for each collision channel
- Easily extensible to more components
- CUDA implementation of the homogeneous mixture solver
- One thread per particle
- Supports species-dependent masses and cross sections
- Designed for large-scale systems


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
