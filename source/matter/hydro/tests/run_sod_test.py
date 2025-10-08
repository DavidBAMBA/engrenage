#!/usr/bin/env python3
"""
Ejecutar solo el test de Sod con comparación de reconstructores.
"""

import sys
import os

# Añadir el path del directorio source desde tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Importar el test
from test import test_riemann_sod

if __name__ == "__main__":
    print("Ejecutando test de Sod con múltiples reconstructores...")
    result = test_riemann_sod()
    print(f"\nResultado final: {'ÉXITO' if result else 'FALLO'}")