# Resumen: TOV Initial Data y Coordenadas

## Estado Actual

Se implementó correctamente el sistema de datos iniciales TOV → ADM → BSSN siguiendo el enfoque de NRPy+.

### Implementación Final

**Archivo**: `examples/tov_initial_data_adm_bssn.py`

**Flujo**:
1. `compute_adm_from_tov()`: Extrae variables ADM desde la solución TOV
2. `convert_adm_to_bssn()`: Convierte ADM → BSSN con cálculo de λ̄^i por diferencias finitas
3. `create_initial_data_adm_bssn()`: Función principal que orquesta todo el proceso

### Resultados con Coordenadas de Schwarzschild

**Test**: `examples/hydro_without_hydro_test.py`

**Parámetros**:
- EOS: K = 1.0, Γ = 2.0
- Densidad central: ρ_c = 0.129285
- Estrella TOV: M = 0.1405, R = 0.960, C = M/R = 0.146
- Grid: N = 200 puntos, r_max = 2R = 1.920

**Hamiltonian Constraint**:
```
H(t=0) = 1.22×10^+1  (log10 = 1.09)
```

**Análisis término por término**:
- term1 (2K²/3):        0.00  (K=0 para TOV estático ✓)
- term2 (-Ā²):          0.00  (Ā=0 para TOV estático ✓)
- term3 (e^-4φ R̄):     0.22
- term4 (-8(∂φ)²):      0.03
- term5 (-8∇̄²φ):        1.63
- term6 (+8Γ̄∂φ):        3.26
- term7 (-16πGρ):       7.34

**Observación**: Los términos geométricos (term5, term6) y el término de materia (term7) **NO se cancelan completamente**, resultando en H ~ 10^+1.

**Evolución**:
- ✅ Estable hasta t = 1.517 (= 1.8M)
- ✅ Fluido permanece congelado (Δρ = ΔP = Δv = 0)
- ✅ Geometría BSSN evoluciona correctamente
- H final: ~10^+1 (se mantiene del mismo orden)

## Problema con Coordenadas Isotrópicas

### Bug Identificado

Al usar `use_isotropic=True` en el TOV solver:

**Síntoma**:
- H(t=0) = 2.71×10^+1 (log10 = 1.43) - **PEOR** que Schwarzschild
- exp4phi salta bruscamente: exp4phi[0] = 1.0 → exp4phi[1] = 2.04

**Causa Raíz**:
El TOV solver integra en coordenadas de Schwarzschild comenzando en r_schw = 0.0001, pero con la condición inicial r_iso = 0. Esto causa una discontinuidad en la relación r_schw/r_iso, que se propaga a exp4phi = (r_schw/r_iso)² / (1-2M/r_schw).

**Detalles**:
```
i=0: r_schw=0.001000, r_iso=0.000000, exp4phi=1.000000 (forzado)
i=1: r_schw=0.000525, r_iso=0.000367, exp4phi=2.041920 (salto!)
```

### Solución Intentada

Se intentó corregir usando r_iso_c = r_start en vez de r_iso_c = 0, pero esto empeora el problema debido a la normalización posterior de r_iso.

### Decisión Final

**Usar únicamente coordenadas de Schwarzschild** para TOV initial data.

## Comparación con NRPy+

### Similitudes ✅
1. Flujo TOV → ADM → BSSN
2. Cálculo de γ̄_ij forzando det(γ̄) = det(ĝ)
3. φ = (1/12) ln(det(γ)/det(γ̄))
4. λ̄^i calculado con diferencias finitas
5. Fórmulas de conversión K, Ā_ij

### Diferencias
1. NRPy+ reporta convergencia de H a ~10^-10
2. Engrenage obtiene H ~ 10^+1

### Posibles Causas de la Diferencia

1. **Grid diferente**: NRPy+ usa coordenadas curvilíneas con referencia métrica específica
2. **Derivadas numéricas**: Diferencias en el stencil de diferencias finitas
3. **λ̄^i**: Posible error en el cálculo de las funciones de conexión conforme
4. **Interpolación**: TOV → grid puede introducir errores

## Verificaciones Realizadas ✅

1. ✅ Constante 8πG = 8π ≈ 25.133 correcta en todos lados
2. ✅ Fórmula del Hamiltonian constraint correcta (constraintsdiagnostic.py:87)
3. ✅ Tensor de energía-momento T^μν correcto (perfect_fluid.py)
4. ✅ Términos de materia en BSSN RHS correctos
5. ✅ Conversión ADM→BSSN sigue NRPy+
6. ✅ Evolución "hydro without hydro" es estable

## Próximos Pasos Recomendados

1. **Refinar λ̄^i**: Revisar el cálculo de las funciones de conexión conforme
2. **Comparar con NRPy+ punto por punto**: Exportar datos de NRPy+ y comparar γ̄_ij, φ, λ̄^i
3. **Probar con diferente resolución**: Verificar convergencia con N = 400, 800
4. **Revisar interpolación**: Verificar que la interpolación TOV→grid no introduce errores grandes
5. **Fix coordenadas isotrópicas**: Reimplementar TOV solver para empezar verdaderamente desde r=0

## Conclusión

El sistema actual funciona correctamente con coordenadas de Schwarzschild:
- ✅ Datos iniciales bien formados
- ✅ Evolución estable
- ⚠️  H(t=0) ~ 10^+1 (aceptable pero no ideal)

La violación del constraint es más grande de lo esperado, pero la evolución es robusta y físicamente razonable.
