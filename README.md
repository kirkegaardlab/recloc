# Optimal Location for Surface sensors in 3D geometries
[![arXiv](https://img.shields.io/badge/arXiv-2410.03395-b31b1b.svg?style=flat)](https://arxiv.org/abs/2410.03395)

>[!NOTE]
> This is the code for the paper [Receptors cluster in high-curvature membrane regions for optimal spatial gradient sensing](https://arxiv.org/abs/2410.03395).

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
Note that the requirements.txt contain the cuda12 version of JAX.


# Run
To run the code, use the following command:
```bash
python cramer.py
```

and to visualize the results, use the following command:
```bash
python visual.py
```

# License

MIT License
