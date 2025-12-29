
import os
import numpy as np
import torch
import gzip
import torch
import gzip
from types import SimpleNamespace
from lib.dss.dss import DSSClient
from lib.dss.gslib import Gslib


def create_dss_acoustic_impedance():
    # Configuration
    project_path = os.getcwd()
    in_folder = "lib/dss/assets"
    out_dir = "data"
    
    facies_path = 'data/facies.npy.gz'
    output_path = 'data/acoustic_impedance.npy.gz'

    # Load Facies
    print(f"Loading facies from {facies_path}...")
    with gzip.open(facies_path, 'rb') as f:
        facies = np.load(f)
    
    print(f"Facies shape: {facies.shape}")
    
    # Facies shape is (N, 1, H, W)
    nx = facies.shape[-1]
    nz = facies.shape[-2]

    # Mock Arguments for DSSClient
    args = SimpleNamespace(
        project_path=project_path,
        in_folder=in_folder,
        outdir=out_dir,
        nx=nx,
        nz=nz,
        var_N_str=[1, 1], # default from generate_dss_realization.py
        var_nugget=[0, 0],
        var_type=[[1], [1]],
        var_ang=[[0, 0], [0, 0]],
        var_range=[[[30, 10]], [[40, 40]]],
        null_val=-9999.99,
        type_of_FM='fullstack',
        ip_type=1, # 1 means run_dss
        n_sim_ip=1 # 1 simulation average
    )
    
    print("Generating hard data with user parameters...")
    n_points = 2000
    mean0, std0 = 2500.0, 200.0
    mean1, std1 = 3500.0, 300.0
    
    os.makedirs(os.path.join(project_path, in_folder), exist_ok=True)

    x0 = np.random.randint(0, nx, n_points)
    y0 = np.random.randint(0, 1, n_points)
    z0 = np.random.randint(0, nz, n_points)
    ip0 = np.random.normal(mean0, std0, n_points)
    data0 = np.stack([x0, y0, z0, ip0], axis=1)
    
    x1 = np.random.randint(0, nx, n_points)
    y1 = np.random.randint(0, 1, n_points)
    z1 = np.random.randint(0, nz, n_points)
    ip1 = np.random.normal(mean1, std1, n_points)
    data1 = np.stack([x1, y1, z1, ip1], axis=1)
    
    Gslib().Gslib_write('Ip_zone0', ['x','y','z','Ip'], data0, 4, 1, n_points, os.path.join(project_path, in_folder))
    Gslib().Gslib_write('Ip_zone1', ['x','y','z','Ip'], data1, 4, 1, n_points, os.path.join(project_path, in_folder))
    
    ipmin = min(ip0.min(), ip1.min())
    ipmax = max(ip0.max(), ip1.max())
    ipzones = {
        0: np.array([ip0.min(), ip0.max()]),
        1: np.array([ip1.min(), ip1.max()])
    }
    
    # Create output directory for DSS
    os.makedirs(os.path.join(out_dir, 'dss'), exist_ok=True)
    
    # Initialize DSS Client
    EM = DSSClient(args, real_fac_model=None, ipmin=ipmin, ipmax=ipmax, ipzones=ipzones)
    
    print("Running DSS simulation for all samples...")
    
    facies_tensor = torch.tensor(facies, dtype=torch.float32)
    if facies_tensor.ndim == 3:
        facies_tensor = facies_tensor.unsqueeze(1)
        
    simulations = EM.run_dss(facies_tensor, 0, args)
    
    res_numpy = simulations.detach().cpu().numpy()
    if res_numpy.ndim == 4 and res_numpy.shape[1] == 1:
        res_numpy = res_numpy.squeeze(1) # (N, H, W)
        
    print(f"Generated data shape: {res_numpy.shape}")
    
    with gzip.open(output_path, 'wb') as f:
        np.save(f, res_numpy)
        
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    create_dss_acoustic_impedance()
