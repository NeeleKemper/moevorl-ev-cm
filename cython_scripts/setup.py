from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        ['electric_vehicle_cy.pyx', 'charging_point_cy.pyx', 'charging_park_cy.pyx', 'environment_cy.pyx'],
        annotate=True),
    include_dirs=[np.get_include()],
)
