# Local Clustering and Global Spreading of Receptors for Optimal Spatial Gradient Sensing

[![Physical Review Letters (PRL)](https://img.shields.io/badge/PRL-10.1103/PhysRevLett.134.158401-990000?style=flat)](https://doi.org/10.1103/PhysRevLett.134.158401)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03395-b31b1b.svg?style=flat)](https://arxiv.org/abs/2410.03395)

This repository contains the code used in our [Physical Review Letters](https://doi.org/10.1103/PhysRevLett.134.158401) paper:

> **Local Clustering and Global Spreading of Receptors for Optimal Spatial Gradient Sensing**  
> Albert Alonso, Robert G. Endres, J. B. Kirkegaard

<p align="center">
  <img src="https://github.com/user-attachments/assets/6608f412-981e-4d9a-8244-e5aa2f052857" height="450" />
</p>

# Setup
Create a virtual enviorment and install the required packages using the following commands:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
>[!NOTE]
> requirements.txt installs the CUDA 12-compatible version of JAX. Modify this if you're using CPU-only or a different CUDA version.

# Run
Run the core simulation:
```bash
python cramer.py
```

Visualize the results:
```bash
python visual.py
```

# License
This code is released under the MIT License.
