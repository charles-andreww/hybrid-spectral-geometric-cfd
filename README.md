#  Accelerating Two-Dimensional Computational Fluid Dynamics via Hybrid Spectral-Geometric Neural Architectures

[![Preprint](https://img.shields.io/badge/Preprint-Research_Square-green?style=flat-square)](https://www.researchsquare.com/article/rs-10172708/v1)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--3187--1544-A6CE39?style=flat-square&logo=orcid)](https://orcid.org/0009-0000-3187-1544)

An open-source PyTorch implementation of the hybrid deep learning architecture designed to accelerate 2D fluid dynamics simulations over highly irregular boundaries. 

By combining **Geometric Routing** (via Signed Distance Fields) with **Spectral Fourier Convolutions**, this network resolves global Navier-Stokes physics instantly without iterative time-stepping.

---

## 📂 Repository Structure

* `src/model_v3.py`: Contains the `GeometricRouting`, `SpectralFourier2D`, and `HybridUNet` PyTorch classes.
* `src/train.py`: Implementation of the custom `PINNLoss` for Deep Supervision mass-conservation grading.
* `src/dataset_v3.py`: Custom PyTorch Dataset loader handling dynamic multi-scale target generation.
* `src/data_v2.py`: The physics engine data generator utilizing Phi-Flow.
* `src/test.py`: Live inference evaluation and Matplotlib metrics comparison.
  
## Quick Start

Clone the repository and install the required physics and deep learning dependencies:

```bash
git clone https://github.com/charles-andreww/hybrid-spectral-geometric-cfd.git
cd hybrid-spectral-geometric-cfd
pip install -r requirements.txt
```
To run the live evaluation on your local machine:
```bash
python src/test.py
```
## Dataset format
The project relies on compressed NumPy array structures. Each steady-state configuration contains the following keys:
```python
import numpy as np

# Load a single wind tunnel simulation
data = np.load('data/sim_v2_0000.npz')

sdf_matrix = data['sdf']  # Shape: (64, 64) -> Continuous geometric boundary mask
u_velocity = data['u']    # Shape: (64, 64) -> Horizontal wind field
v_velocity = data['v']    # Shape: (64, 64) -> Vertical wind field

print(f"Loaded jagged obstacle geometry with shape: {sdf_matrix.shape}")
```
**Full Dataset Archive**: The complete set of 3,000 highly resolved simulation files used to train this network is openly hosted at: https://huggingface.co/datasets/Charles-Andrew/hybrid-spectral-geometric-cfd.

## Citation
```
Angulo Rosales, Carlos Andrés. (2026). Accelerating Two-Dimensional Computational Fluid Dynamics via Hybrid Spectral-Geometric Neural Architectures. 10.21203/rs.3.rs-10172708/v1. 
```
