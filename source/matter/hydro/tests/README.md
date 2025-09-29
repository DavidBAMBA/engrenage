# Hydrodynamics Test Suite

This directory contains all test files for the relativistic hydrodynamics module in Engrenage.

## Test Files

### Core Conversion Tests
- **`test_cons2prim_comprehensive.py`** - Comprehensive cons2prim conversion test suite
  - Performance benchmarks
  - Correctness validation
  - Failure analysis
  - Vectorized vs legacy comparison
  - Edge case testing

### Physical Validation Tests
- **`test.py`** - Hydrodynamics validation tests
  - Sod shock tube test
  - Spherical blast wave tests (weak/strong)
  - Conservation tests
  - Riemann solver validation

### Specialized Tests
- **`test-valencia.py`** - Valencia formulation with reference metric tests
- **`tov_test.py`** - TOV star tests for BSSN-Hydro coupling validation

## Running Tests

### Individual Tests
```bash
# From project root
cd tests/hydro

# Comprehensive cons2prim test suite
python test_cons2prim_comprehensive.py

# Hydrodynamics validation (Sod, blast waves)
python test.py

# TOV star test
python tov_test.py
```

### Using pytest
```bash
# From project root
pytest tests/hydro/

# Specific test
pytest tests/hydro/test_cons2prim_comprehensive.py
```

## Expected Results

### test_cons2prim_comprehensive.py
- **Success rate**: 100% for all physical test cases
- **Energy conservation**: Errors < 10⁻¹¹
- **Performance**: 300-400x speedup vs legacy implementation
- **Scaling**: 10-20 μs per point for large grids (N > 1000)

### test.py
- **Sod shock tube**: Clean shock, contact discontinuity, rarefaction wave
- **Blast waves**: Proper spherical expansion with correct profiles
- **Conservation**: Mass, momentum, energy conserved to machine precision

### tov_test.py
- **Stability**: TOV star should remain stable over evolution
- **BSSN-Hydro coupling**: Consistent metric and fluid evolution
- **Accuracy**: Deviation from analytic solution < 1e-12

## Test Output

Tests generate diagnostic plots:
- `cons2prim_performance.png` - Performance scaling plots
- `sod_final2.png` - Sod shock tube results
- `blast_final_weak2.png` / `blast_final_strong2.png` - Blast wave results
- `tov_*.png` - TOV star evolution plots

## Troubleshooting

### Import Errors
Ensure you're running from the correct directory and that the source path is properly added:
```python
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
```

### Performance Issues
For performance tests, ensure:
- No other heavy processes running
- Sufficient memory available
- NumPy using optimized BLAS libraries

### Test Failures
If tests fail:
1. Check that all dependencies are installed (`requirements.txt`)
2. Verify source code is up to date
3. Check for any local modifications to the physics modules