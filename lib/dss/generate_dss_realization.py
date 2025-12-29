
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from .dss import DSSClient
from .gslib import Gslib


def main():
    parser = argparse.ArgumentParser(description="Generate DSS realizations from a 2D facies image.")
    
    # Input/Output
    parser.add_argument("--facies_file", type=str, required=True, help="Path to the 2D facies image (npy file expected). 0 for facies 0, 1 for facies 1.")
    parser.add_argument("--project_path", type=str, default=os.getcwd(), help="Base project path.")
    parser.add_argument("--in_folder", type=str, default="assets", help="Input folder containing DSS executable and where intermediate files will be written.")
    parser.add_argument("--outdir", type=str, default="./tmp", help="Output directory for results.")
    
    # Distribution Properties
    parser.add_argument("--mean0", type=float, default=2500.0, help="Mean Ip for Facies 0.")
    parser.add_argument("--std0", type=float, default=200.0, help="Std Dev Ip for Facies 0.")
    parser.add_argument("--mean1", type=float, default=3500.0, help="Mean Ip for Facies 1.")
    parser.add_argument("--std1", type=float, default=300.0, help="Std Dev Ip for Facies 1.")
    
    # Simulation Parameters
    parser.add_argument("--nx", type=int, default=256, help="Number of pixels in X.")
    parser.add_argument("--nz", type=int, default=256, help="Number of pixels in Z (or Y in 2D).")
    parser.add_argument("--n_sim_ip", type=int, default=1, help="Number of Ip simulations to average.")
    
    # Variogram Parameters
    parser.add_argument("--var_N_str", default=[1,1], type=int, nargs='+', help='number of variogram structures per facies')
    parser.add_argument("--var_nugget", default=[0,0], type=float, nargs='+', help='variogram nugget per facies')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.project_path + args.in_folder, exist_ok=True)
    
    # 1. Load Facies Image
    print(f"Loading facies from {args.facies_file}...")
    try:
        if args.facies_file.endswith('.npy'):
            facies = np.load(args.facies_file)
        else:
            print("Error: Only .npy format supported for facies_file currently.")
            return
    except Exception as e:
        print(f"Failed to load facies file: {e}")
        return

    # Ensure facies is the right shape and type
    if facies.ndim == 2:
        facies = facies[np.newaxis, np.newaxis, :, :] # (1, 1, nz, nx)
    elif facies.ndim == 3:
        facies = facies[np.newaxis, :, :, :]
    
    # 2. Generate Ip_zone files
    print("Generating synthetic hard data (Ip_zone files)...")
    n_samples = 1000
    
    # Facies 0
    ip0_values = np.random.normal(args.mean0, args.std0, n_samples)
    ip0_df = pd.DataFrame({
        'x': np.random.randint(0, args.nx, n_samples),
        'y': np.random.randint(0, 1, n_samples), # 2D case, y is 1
        'z': np.random.randint(0, args.nz, n_samples),
        'Ip': ip0_values
    })
    
    # Facies 1
    ip1_values = np.random.normal(args.mean1, args.std1, n_samples)
    ip1_df = pd.DataFrame({
        'x': np.random.randint(0, args.nx, n_samples),
        'y': np.random.randint(0, 1, n_samples),
        'z': np.random.randint(0, args.nz, n_samples),
        'Ip': ip1_values
    })
    
    # Write to Gslib format
    Gslib().Gslib_write('Ip_zone0', ['x','y','z','Ip'], ip0_df, 4, 1, n_samples, args.project_path + args.in_folder)
    Gslib().Gslib_write('Ip_zone1', ['x','y','z','Ip'], ip1_df, 4, 1, n_samples, args.project_path + args.in_folder)
    
    print(f"Hard data written to {args.project_path + args.in_folder}")

    # 3. Setup ElasticModels
    args.var_type = [[1], [1]] # Default spherical
    args.var_ang = [[0,0], [0,0]]
    args.var_range = [[[30,10]], [[40,40]]] # Default ranges
    args.null_val = -9999.99
    args.type_of_FM = 'fullstack'
    args.ip_type = 1 # 1 means run_dss, 0 means deterministic
    
    # Calculate min/max for the zones
    ipmin = min(ip0_values.min(), ip1_values.min())
    ipmax = max(ip0_values.max(), ip1_values.max())
    
    ipzones = {
        0: np.array([ip0_values.min(), ip0_values.max()]),
        1: np.array([ip1_values.min(), ip1_values.max()])
    }
    
    print("Initializing ElasticModels...")
    EM = DSSClient(args, real_fac_model=None, ipmin=ipmin, ipmax=ipmax, ipzones=ipzones)
    
    # 4. Run DSS
    print("Running DSS...")
    facies_tensor = torch.tensor(facies, dtype=torch.float32)
    
    # run_dss expects (facies_mod, i, args)
    # The 'i' argument is used for iteration count in the original script to change kriging type.
    # i < 1 -> Simple Kriging (krig_type=0)
    # i >= 1 -> Local Collocated Cokriging (krig_type=5)
    # If i < 1, sec_var_file='No file'
    simulated_ip = EM.run_dss(facies_tensor, 0, args)
    
    # 5. Save and Plot Results
    print("Simulation complete.")
    
    # Extract the result
    # simulated_ip shape: [Batch, 1, Z, X]
    res_numpy = simulated_ip.detach().cpu().numpy().squeeze()
    
    output_file = os.path.join(args.outdir, "generated_ip.npy")
    np.save(output_file, res_numpy)
    print(f"Saved result to {output_file}")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Input Facies")
    plt.imshow(facies.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Acoustic Impedance")
    plt.imshow(res_numpy, cmap='jet')
    plt.colorbar()
    
    plot_file = os.path.join(args.outdir, "result_plot.png")
    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")
    plt.close()

if __name__ == "__main__":
    main()
