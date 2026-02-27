import os
import re
import shutil
import random
import subprocess
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp


events =    10
Npart =     4000
Nt =        50_000


def COMPILE_C(solver):
    return [
                "gcc", solver,
                "-O3", "-lm",
                "-o", "solver.out"
            ]

def COMPILE_CUDA(solver):
    return [
                "nvcc", solver,
                "-O3", "-std=c++14",
                "-gencode", "arch=compute_75,code=sm_75",
                "-gencode", "arch=compute_75,code=compute_75",
                "-o", "solver.out"
            ]

solver = "mixture.c"
SRC_CU = "/content/boltzmann_transport/" + solver

def run_single_event(args):
    x1, T, s11, s22, s12, m1, m2, observable = args
    x2 =  1.0 - x1
    workdir = tempfile.mkdtemp(prefix=observable + "_")

    try:
        # copy CUDA source from absolute path
        shutil.copy(SRC_CU, os.path.join(workdir, solver))

        # switch to isolated directory
        os.chdir(workdir)

        with open(solver, "r") as f:
            content = f.readlines()

        seed = random.randint(0, 1_000_000)

        # edit defines (1-based → 0-based)
        content[19]  = f"#define SEED    {seed}\n"
        content[10] = f"#define N1    {int(Npart * x1)}\n"
        content[11] = f"#define N2    {int(Npart * x2)}\n"

        content[12] = f"#define MASS1    {m1}\n"
        content[13] = f"#define MASS2    {m2}\n"

        content[14] = f"#define SIGMA11    {s11}\n"
        content[15] = f"#define SIGMA22    {s22}\n"
        content[16] = f"#define SIGMA12    {s12}\n"

        content[21] = f"#define TEMPERATURE     {T}\n"
        content[25] = f"#define NT    {Nt}\n"
        content[26] = f"#define TMAX    {200}\n"
        content[27] = f"double DT = 0.001;\n"

        if observable == "shear":
            content[355] = "observable[t] = shear_stress_tensor_xy(&sys);\n"
        elif observable == "bulk":
            content[355] = "observable[t] = bulk_viscous_pressure(&sys);\n"

        content[353] = "if (t == 100 && DT < 1e7) {if (coll_count/(double)NPART < 5) {t = 0;DT *= 2.0;coll_count = 0;}}\n"


        with open(solver, "w") as f:
            f.writelines(content)

        subprocess.check_call(
            COMPILE_C(solver),
            stderr=subprocess.DEVNULL,
        )

        output = subprocess.check_output(["./solver.out"]).decode()
        match = re.search(r"Eta\s*=\s*([0-9eE.+-]+)", output)

        if not match:
            raise ValueError("Eta not found in output")
        eta = float(match.group(1))

        # Load correlator
        df = pd.read_csv("output.csv")

        return df, eta

    finally:
        os.chdir("/")
        shutil.rmtree(workdir)

def average_correlators(dfs):
    # assume identical time grids
    time = dfs[0]["time"].values

    corr_stack = np.stack(
        [df["correlator"].values for df in dfs],
        axis=0
    )

    mean_corr = corr_stack.mean(axis=0)
    err_corr  = corr_stack.std(axis=0) / np.sqrt(len(dfs))

    return time, mean_corr, err_corr


def plot_avg_corr(time, corr, err):
    plt.figure()
    plt.plot(time, corr, label="Mean correlator")
    plt.fill_between(
        time,
        corr - err,
        corr + err,
        alpha=0.3,
        label="Stat. error"
    )

    plt.xlabel(r"$t\;\mathrm{[GeV^{-1}]}$")
    plt.ylabel(
        r"$C(t);\mathrm{[GeV^{8}]}$"
    )
    plt.legend()
    plt.grid(True)
    plt.show()

x1 = 0.5 
T = 0.5
mass1 = 0.5
mass2 = 0.5
s11 = 0.1
s22 = 0.3
s12 = 0.6
observable = "shear"
err = None
eta = None

def main():
    global eta, err
    start = time.time()
    with mp.Pool(processes=events) as pool:
        args = [(x1, T, s11,s22,s12, mass1, mass2, observable) for _ in range(events)]
        output = pool.map(run_single_event, args)
        dfs, res = zip(*output)

        dfs = list(dfs)
        res = np.array(res)

        t, mean_corr, err_corr = average_correlators(dfs)

        plot_avg_corr(t, mean_corr, err_corr)

        res = np.array(res)
        eta = np.mean(res)
        err  = np.std(res, ddof=1) / np.sqrt(events)

    end = time.time()
    print(f"x1={x1}, m1={mass1},m2={mass2}, T={T}, s11={s11},s22={s22},s12={s12}, eta={eta:.6f}, err={err:.6f}")
    print(f"Time: {end-start} s")
    return eta, err


if __name__ == "__main__":
    main()
