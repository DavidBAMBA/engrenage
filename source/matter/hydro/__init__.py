# matter/hydro/__init__.py

"""
Relativistic hydrodynamics module for engrenage.

Valencia formulation with reference metric (spherical). This __init__ is kept
LIGHTWEIGHT on purpose: we avoid heavy imports at package import time so that
tools like `python -m source.matter.hydro.test` don't fail on transitive deps.
"""

from importlib import import_module

__all__ = [
    'PerfectFluid',
    'ValenciaReferenceMetric',
    'IdealGasEOS',
    'PolytropicEOS',
    'ConservativeToPrimitive',   # may be absent; exposed lazily if present
    'HLLERiemannSolver',
    'MinmodReconstruction',
    'create_perfect_fluid',
    'EOS_TYPES',
    'RECONSTRUCTION_TYPES',
    'RIEMANN_SOLVERS',
]

# -------- Lazy attribute loading (PEP 562) --------
def __getattr__(name):
    if name == 'PerfectFluid':
        return import_module('.perfect_fluid', __name__).PerfectFluid
    if name == 'ValenciaReferenceMetric':
        return import_module('.valencia_reference_metric', __name__).ValenciaReferenceMetric
    if name == 'IdealGasEOS':
        return import_module('.eos', __name__).IdealGasEOS
    if name == 'PolytropicEOS':
        return import_module('.eos', __name__).PolytropicEOS
    if name == 'HLLERiemannSolver':
        return import_module('.riemann', __name__).HLLERiemannSolver
    if name == 'MinmodReconstruction':
        return import_module('.reconstruction', __name__).MinmodReconstruction
    if name == 'ConservativeToPrimitive':
        # Opcional: si existe una clase así, exponla; si no, no rompas.
        mod = import_module('.cons2prim', __name__)
        return getattr(mod, 'ConservativeToPrimitive', None)
    if name == 'EOS_TYPES':
        eos = import_module('.eos', __name__)
        return {'ideal': eos.IdealGasEOS, 'polytropic': eos.PolytropicEOS}
    if name == 'RECONSTRUCTION_TYPES':
        rec = import_module('.reconstruction', __name__)
        return {'minmod': rec.MinmodReconstruction}
    if name == 'RIEMANN_SOLVERS':
        r = import_module('.riemann', __name__)
        return {'hlle': r.HLLERiemannSolver}
    if name == 'create_perfect_fluid':
        # Devuelve la factory como función cerrada para que los imports sean internos
        def _factory(gamma=1.4, spacetime_mode="fixed_minkowski",
                    atmosphere_rho=1e-13, reconstruction="minmod",
                    riemann_solver="hlle"):
            eos_mod  = import_module('.eos', __name__)
            rec_mod  = import_module('.reconstruction', __name__)
            rie_mod  = import_module('.riemann', __name__)
            pf_mod   = import_module('.perfect_fluid', __name__)

            # EOS
            if reconstruction != "minmod":
                raise ValueError(f"Unknown reconstruction method: {reconstruction}")
            if riemann_solver != "hlle":
                raise ValueError(f"Unknown Riemann solver: {riemann_solver}")

            eos = eos_mod.IdealGasEOS(gamma)
            reconstructor = rec_mod.MinmodReconstruction()
            riemann = rie_mod.HLLERiemannSolver()

            fluid = pf_mod.PerfectFluid(
                eos=eos,
                spacetime_mode=spacetime_mode,
                atmosphere_rho=atmosphere_rho,
                reconstructor=reconstructor,
                riemann_solver=riemann
            )
            return fluid
        return _factory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
