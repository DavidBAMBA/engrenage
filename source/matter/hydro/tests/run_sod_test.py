#!/usr/bin/env python3
"""
Ejecutar solo el test de Sod con comparación de reconstructores.
"""

import sys
import os

# Añadir el path del directorio source desde tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Importar el test
from test import test_blast_compare, test_riemann_sod

if __name__ == "__main__":
    print("Ejecutando test de Sod con múltiples reconstructores...")
    #result = test_blast_compare(case="weak")
    result = test_riemann_sod()
    #result2 = test_blast_compare(case="strong")
    #print(f"\nResultado final: {'ÉXITO' if result and result2 else 'FALLO'}")