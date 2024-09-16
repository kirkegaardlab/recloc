# Optimal Location for Surface sensors in 3D geometries
>[!NOTE]
> This is the code for the paper "Optimal Cell-Surface Receptor Location for Spatial Gradient Sensing".

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b5d425a-2791-46f2-bea5-07fcb39ba3e6" height="420" />
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

# Project structure

The different experiments (setups) for the figures are on the [experiments](experiments/) directory.

# License

MIT License
