import subprocess
import numpy as np
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor
import random
import os

import glob
import matplotlib.pyplot as plt

# --- CONFIG ---
EXECUTABLE = "sim.exe"
NUM_RUNS = 20  # number of parallel simulations
PATH = "./results/"

# --- PARSER ---
def parse_output(output, seed):
    eta_match = re.search(r"Eta:\s*([0-9.eE+-]+)", output)
    zeta_match = re.search(r"Zeta:\s*([0-9.eE+-]+)", output)

    if not eta_match or not zeta_match:
        raise ValueError(f"Failed to parse output for seed {seed}:\n{output}")

    eta = float(eta_match.group(1))
    zeta = float(zeta_match.group(1))
    return eta, zeta

# --- RUN ONE SIMULATION ---
def run_sim(args):
    seed, input_file = args
    try:
        result = subprocess.run(
            [EXECUTABLE, input_file],  # pass the filename as argument,
            capture_output=True,
            text=True,
            shell=True
        )
        return parse_output(result.stdout, seed)
    except Exception as e:
        print(f"Simulation with seed {seed} failed!")
        print(result.stdout)
        raise e

# --- RUN NUM_RUNS PARALLEL SIMS ---
def run_batch(seeds):
    # each seed has its own input file: input_1.txt, input_2.txt, ...
    args_list = [(seed, PATH + f"input_{i+1}.txt") for i, seed in enumerate(seeds)]

    with ProcessPoolExecutor(max_workers=NUM_RUNS) as executor:
        results = list(executor.map(run_sim, args_list))

    etas = np.array([r[0] for r in results])
    zetas = np.array([r[1] for r in results])

    return etas.mean(), etas.std(), zetas.mean(), zetas.std()

# --- COMPILE STEP ---
def compile_code():
    subprocess.run(
        "nvcc mixture_cuda.cu -o sim.exe",
        shell=True,
        check=True
    )

# --- MODIFY INPUT FILE ---
def modify_input(row, seeds):
    repeat = 7
    for i, seed in enumerate(seeds):
        filename = PATH + f"input_{i+1}.txt"
        with open(filename, "w") as f:
            f.write(f"seed         {seed}\n")
            f.write(f"x1           {row['x1']}\n")
            f.write(f"T            {row['T']}\n")
            f.write(f"s11          {row['s11']}\n")
            f.write(f"s22          {row['s22']}\n")
            f.write(f"s12          {row['s12']}\n")
            f.write(f"mass1        {row['mass1']}\n")
            f.write(f"mass2        {row['mass2']}\n")
            f.write(f"repeat       {repeat}\n")


# --- PLOT MEAN CORRELATOR ---
def average_and_plot(folder_path, pattern="bulk_input_*.csv", label="some caption",
                     out_csv="average.csv", out_plot="average_plot.png"):
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not files:
        raise ValueError("No matching CSV files found.")

    dfs = [pd.read_csv(f) for f in files]

    time = dfs[0]["time"].values
    tensor_stack = np.stack([df["tensor"].values for df in dfs])
    correlator_stack = np.stack([df["correlator"].values for df in dfs])

    tensor_mean = tensor_stack.mean(axis=0)
    correlator_mean = correlator_stack.mean(axis=0)
    correlator_std = correlator_stack.std(axis=0)

    # Save averaged CSV
    avg_df = pd.DataFrame({
        "time": time,
        "tensor": tensor_mean,
        "correlator": correlator_mean
    })
    avg_df.to_csv(os.path.join(folder_path, out_csv), index=False)

    # Plot with std shading
    plt.figure()
    plt.plot(time, correlator_mean, label="Mean")
    plt.fill_between(time,
                     correlator_mean - correlator_std,
                     correlator_mean + correlator_std,
                     alpha=0.3,
                     label="Stat. error")

    plt.xlabel("Time [fm/c]")
    plt.ylabel("Correlator [GeV^8]")
    plt.title(f"{label} ({len(files)} events)", fontsize=10, pad=10)
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(folder_path, out_plot))
    plt.close()

# --- MAIN PIPELINE ---
def process_csv(csv_file):
    df = pd.read_csv(csv_file)

    # compile once
    compile_code()

    for idx, row in df.iterrows():
        
        # Skip already computed rows 
        # if not pd.isna(row["zeta"]): 
        #     print(f"row {idx} already computed")
        #     continue

        print(f"Processing row {idx}...")

        # generate unique seeds for this row
        seeds = [random.randint(1, 10**6) for _ in range(NUM_RUNS)]

        # write input files
        modify_input(row, seeds)

        # run batch
        eta_mean, eta_std, zeta_mean, zeta_std = run_batch(seeds)

        # store results
        df.loc[idx, "eta"] = eta_mean
        df.loc[idx, "eta_err"] = eta_std
        df.loc[idx, "zeta"] = zeta_mean
        df.loc[idx, "zeta_err"] = zeta_std

        # save after each row
        df.to_csv(csv_file, index=False, float_format="%.8g")

        label = ",".join(f"{col}={df.loc[idx, col]:.2e}" for col in df.columns)

        # save the average correlator data
        average_and_plot(PATH, pattern="bulk_input_*.csv", label=label,
                     out_csv=f"avg_bulk_{idx}.csv", out_plot=f"avg_bulk_plot_{idx}.png")
        
        average_and_plot(PATH, pattern="shear_input_*.csv", label=label,
                     out_csv=f"avg_shear_{idx}.csv", out_plot=f"avg_shear_plot_{idx}.png")

    print("Done.")

# --- RUN ---
if __name__ == "__main__":
    process_csv("Data.csv")
