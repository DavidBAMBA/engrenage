#!/usr/bin/env python3
"""
Ejecutar solo el test de Blast Wave (strong) con MP5.
"""

import sys
import os

# Añadir el path del directorio source desde tests
sys.path.insert(0, '/home/yo/repositories/engrenage')

# Importar el test
from test import test_blast_wave

if __name__ == "__main__":
    print("Ejecutando test de Strong Blast Wave...")
    result = test_blast_wave(case='strong')
    print(f"\nResultado final: {'ÉXITO' if result else 'FALLO'}")
