
from singlet import SigmaCurve, OmegaLens
from surface_generation import revolve_curve
from viewer import show_vispy, info_view

import json
from vispy import app
import matplotlib.pyplot as plt

# Parámetros de discretización y visualización
N_RHO = 10
N_ANG = 10
SHOW_3D = True
SHOW_2D = True

# Archivo de parámetros
FILE_NAME = 'queue/params.json'
with open(FILE_NAME, "r") as f:
    params = json.load(f)

# Lectura de parámetros ópticos
z0 = params['z0']
z1 = params['z1']
z2 = params['z2']
n0 = params['n0']
n1 = params['n1']
t = params.get('t', 0.0)

# Construcción de curvas Sigma
sigma1 = SigmaCurve(z0, z1, n0, n1, rho_points=N_RHO)
sigma2 = SigmaCurve(-z1, z2, n1, n0, rho_points=N_RHO, t_shift=t)

# Lente a partir de las dos curvas
lens = OmegaLens(sigma1, sigma2)
intersection_point = lens.intersection_point
curve = lens.get_points()

print(f'Intersection point at (z,r) = {intersection_point}')
print(f'===== Lens Data =====')
print(f'Object and image positions z0 = {z0} cm, zi = {z2} cm')
print(f'width = {t} cm')
print(f'diameter = {2 * intersection_point[1]} cm')

# Malla 3D de la lente
vertices, faces = revolve_curve(curve, N_ANG)

if SHOW_3D:
    show_vispy(vertices, faces, SHOW_WIREFRAME=True)

if SHOW_2D:
    # Vista 3D
    info_view(
        sigma1.get_points(),
        sigma2.get_points(),
        curve,
        intersection_point,
        width=t,
        diameter=2*intersection_point[1]
    )
