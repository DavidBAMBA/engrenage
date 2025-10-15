# -*- coding: utf-8 -*-
"""
PINN para las ecuaciones TOV en 1D (r). Compara contra integración numérica.

Requisitos:
  pip install numpy scipy matplotlib torch

Notas:
- La "referencia" se obtiene con solve_ivp (Radau) y tolerancias muy estrictas.
- La "numérica" para comparar errores usa solve_ivp con tolerancias moderadas.
- El PINN entrena sobre puntos de colisión en [r_start, r_end]; la atmósfera
  se modela con un switch suave sigmoidal para "congelar" d rho0 / dr ~ 0
  cuando rho0 < fluid_atmos.

Autor: (tu nombre)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1) Parámetros físicos y del dominio
# =============================================================================
fluid_gamma = 2.0       # Índice adiabático
fluid_kappa = 1e2       # Constante politrópica
TOV_rho0    = 0.00128   # Densidad central
fluid_atmos = 1e-8      # Corte de atmósfera
smallpi     = np.arccos(-1.0)

r_start = 1e-6          # Evitar singularidad en r=0
r_end   = 50.0          # Radio máximo

# Semillas para reproducibilidad
np.random.seed(1234)
torch.manual_seed(1234)

# =============================================================================
# 2) Sistema TOV (RHS) – igual que tu implementación base
# =============================================================================
def tov_system(r, y):
    A, rho0 = y
    eps = 1e-6
    r_safe = r if r > eps else eps

    e = fluid_kappa / (fluid_gamma - 1.0) * rho0**(fluid_gamma - 1.0)
    rho_total = rho0 * (1.0 + e)

    dA_dr = A * ((1.0 - A)/r_safe + 8.0 * np.pi * r_safe * A * rho_total)

    m = (r_safe / 2.0) * (1.0 - 1.0/A)

    term = (rho0**(1.0 - fluid_gamma) / fluid_kappa + fluid_gamma / (fluid_gamma - 1.0))
    if rho0 < fluid_atmos:
        drho0_dr = 0.0
    else:
        drho0_dr = -rho0 * term * (m / r_safe**2 + 4.0*np.pi*r_safe*fluid_kappa*rho0**fluid_gamma) \
                   / (fluid_gamma * (1.0 - 2.0*m / r_safe))
    return [dA_dr, drho0_dr]

# =============================================================================
# 3) Solución de referencia (muy estricta) y malla para evaluación
# =============================================================================
sol_ref = solve_ivp(tov_system, [r_start, r_end], [1.0, TOV_rho0],
                    method='Radau', dense_output=True,
                    rtol=1e-12, atol=1e-14)

r_ref = np.linspace(r_start, r_end, 5000)
y_ref = sol_ref.sol(r_ref)
A_ref = y_ref[0]
rho0_ref = y_ref[1]

# =============================================================================
# 4) Solución numérica "moderada" para comparar errores
# =============================================================================
def ivp_integration(r_start, r_end, rtol, atol):
    sol = solve_ivp(tov_system, [r_start, r_end], [1.0, TOV_rho0],
                    method='Radau', dense_output=True, rtol=rtol, atol=atol)
    r_dense = np.linspace(r_start, r_end, 5000)
    y_dense = sol.sol(r_dense)
    return r_dense, y_dense[0], y_dense[1]

# Elige una tolerancia "moderada"
rtol_num, atol_num = 1e-10, 1e-12
r_ivp, A_ivp, rho0_ivp = ivp_integration(r_start, r_end, rtol_num, atol_num)

# Errores numéricos vs. referencia (L2 relativo)
def rel_L2(u, uref):
    return np.linalg.norm(u - uref) / np.linalg.norm(uref)

err_num_A   = rel_L2(A_ivp,   A_ref)
err_num_rho = rel_L2(rho0_ivp, rho0_ref)

print(f"[Numérica vs. referencia]  A(r): {err_num_A:.3e},  rho0(r): {err_num_rho:.3e}")

# =============================================================================
# 5) PINN en PyTorch
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Normalización del input (ayuda al entrenamiento)
def normalize_r(r_np):
    # normaliza r a [-1, 1]
    return 2.0 * (r_np - r_start) / (r_end - r_start) - 1.0

# --- Red MLP
class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, width=128, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Construcción del PINN: formas de prueba para imponer BC exactas y monotonicidad
#     A(r) = 1 + (r - r0) * (g_A(r))^2 >= 1, monótonamente no decreciente
#     rho0(r) = rho_c - (r - r0) * (g_rho(r))^2  (no creciente con r)
class TOV_PINN(nn.Module):
    def __init__(self, r0):
        super().__init__()
        self.r0 = r0
        self.mlp = MLP(in_dim=1, out_dim=2, width=128, depth=5)

    def forward(self, r):
        """
        r: tensor shape (N,1) en coordenada física (no normalizada), requiere grad
        """
        # normaliza a [-1,1] para la red
        r_scaled = 2.0 * (r - r_start) / (r_end - r_start) - 1.0
        out = self.mlp(r_scaled)
        gA   = out[:, :1]
        gRho = out[:, 1:]

        A_pred   = 1.0 + (r - self.r0) * torch.square(gA) + 1e-12
        rho_pred = TOV_rho0 - (r - self.r0) * torch.square(gRho)  # puede cruzar cero; trataremos abajo

        return A_pred, rho_pred

# --- Residuos de la física (RHS) en PyTorch
def tov_residuals(r, A_pred, rho_pred, sharp=200.0):
    """
    Calcula residuos de las EDO en forma:
      resA  = dA/dr - A * ((1-A)/r + 8*pi*r*A*rho_total)
      resRo = drho/dr - s * RHS(rho)   con s ~ Heaviside(rho - atmos)

    Usamos rho_phys >= 1e-12 para evitar potencias inválidas.
    """
    eps = 1e-6
    r_safe = torch.clamp(r, min=eps)

    ones = torch.ones_like(A_pred)
    dA_dr   = torch.autograd.grad(A_pred, r, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    drho_dr = torch.autograd.grad(rho_pred, r, grad_outputs=torch.ones_like(rho_pred), create_graph=True, retain_graph=True)[0]

    rho_phys = torch.clamp(rho_pred, min=1e-12)
    e = (fluid_kappa / (fluid_gamma - 1.0)) * torch.pow(rho_phys, fluid_gamma - 1.0)
    rho_total = rho_phys * (1.0 + e)

    resA = dA_dr - A_pred * ((1.0 - A_pred)/r_safe + 8.0 * np.pi * r_safe * A_pred * rho_total)

    # Masa m(r) = (r/2)*(1 - 1/A)
    m = (r_safe / 2.0) * (1.0 - 1.0 / A_pred)

    term = torch.pow(rho_phys, 1.0 - fluid_gamma) / fluid_kappa + fluid_gamma / (fluid_gamma - 1.0)
    RHS = -rho_phys * term * (m / (r_safe**2) + 4.0*np.pi*r_safe*fluid_kappa*torch.pow(rho_phys, fluid_gamma)) \
          / (fluid_gamma * (1.0 - 2.0*m / r_safe + 1e-14))

    # Switch sigmoidal para atmósfera
    s = torch.sigmoid(sharp * (rho_phys - fluid_atmos))

    resRho = drho_dr - s * RHS

    return resA, resRho

# --- Preparar puntos de colisión
N_colls = 6000
r_colls = np.random.rand(N_colls, 1)  # en [0,1]
r_colls = r_start + (r_end - r_start) * r_colls

# Refuerzo de puntos cerca del centro para capturar gradientes fuertes
N_core = 2000
core = r_start + (1e-2) * np.random.rand(N_core, 1)  # [r_start, r_start+1e-2]
r_colls[:N_core, :] = core

r_colls_t = torch.tensor(r_colls, dtype=torch.float64, device=device, requires_grad=True)

# --- Instanciar PINN, optimizadores
torch.set_default_dtype(torch.float64)
model = TOV_PINN(r0=r_start).to(device)

# Optimizadores: primero Adam, luego (opcional) LBFGS
optimizer = optim.Adam(model.parameters(), lr=1e-3)
use_lbfgs = True
lbfgs = optim.LBFGS(model.parameters(), max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12, history_size=50, line_search_fn='strong_wolfe') if use_lbfgs else None

# Ponderaciones de pérdidas
wA, wRho = 1.0, 1.0

# --- Bucle de entrenamiento
def pinn_loss():
    A_pred, rho_pred = model(r_colls_t)
    resA, resRho = tov_residuals(r_colls_t, A_pred, rho_pred, sharp=200.0)
    # MSE de residuos
    loss_pde = wA * torch.mean(resA**2) + wRho * torch.mean(resRho**2)
    # Penalización suave para negatividad de rho_pred (para estabilizar)
    pen_neg_rho = torch.mean(torch.relu(-rho_pred + 1e-14)**2)
    # Penalizar explosiones en A para estabilidad (opcional)
    pen_A = torch.mean(torch.relu(A_pred - 100.0)**2)
    return loss_pde + 1e-4 * pen_neg_rho + 1e-8 * pen_A, loss_pde, pen_neg_rho

# Entrenamiento con Adam
epochs_adam = 8000
for it in range(1, epochs_adam+1):
    optimizer.zero_grad()
    loss, loss_pde, pen = pinn_loss()
    loss.backward()
    optimizer.step()
    if it % 1000 == 0:
        print(f"[Adam {it:5d}] loss={loss.item():.3e}  (pde={loss_pde.item():.3e}, pen={pen.item():.3e})")

# Refinamiento con LBFGS (opcional)
if use_lbfgs:
    def closure():
        lbfgs.zero_grad()
        loss, loss_pde, pen = pinn_loss()
        loss.backward()
        return loss
    print("[LBFGS] Optimización fina...")
    lbfgs.step(closure)
    loss, loss_pde, pen = pinn_loss()
    print(f"[LBFGS] loss_final={loss.item():.3e}  (pde={loss_pde.item():.3e}, pen={pen.item():.3e})")

# =============================================================================
# 6) Evaluación del PINN y comparación de errores
# =============================================================================
with torch.no_grad():
    r_eval_t = torch.tensor(r_ref.reshape(-1,1), dtype=torch.float64, device=device, requires_grad=False)
    A_pinn_t, rho_pinn_t = model(r_eval_t)
    A_pinn   = A_pinn_t.cpu().numpy().reshape(-1)
    rho_pinn = rho_pinn_t.cpu().numpy().reshape(-1)

# Errores relativos L2 del PINN vs referencia
err_pinn_A   = rel_L2(A_pinn,   A_ref)
err_pinn_rho = rel_L2(rho_pinn, rho0_ref)

print(f"[PINN vs. referencia]      A(r): {err_pinn_A:.3e},  rho0(r): {err_pinn_rho:.3e}")

# =============================================================================
# 7) Gráficas comparativas
# =============================================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(r_ref, A_ref, 'k-',  label='Referencia A(r)', lw=2)
plt.plot(r_ivp, A_ivp, 'r-.', label='solve_ivp (moderado)', lw=1.5)
plt.plot(r_ref, A_pinn, 'b--', label='PINN', lw=1.5)
plt.xlabel('r')
plt.ylabel('A(r)')
plt.title('Comparación de A(r)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(r_ref, rho0_ref, 'k-',  label='Referencia ρ₀(r)', lw=2)
plt.plot(r_ivp, rho0_ivp, 'r-.', label='solve_ivp (moderado)', lw=1.5)
plt.plot(r_ref, rho_pinn, 'b--', label='PINN', lw=1.5)
plt.axhline(fluid_atmos, color='gray', ls=':', label='Atmósfera', lw=1)
plt.xlabel('r')
plt.ylabel('ρ₀(r)')
plt.title('Comparación de ρ₀(r)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 8) Resumen de errores
# =============================================================================
print("\n================  RESUMEN DE ERRORES (L2 relativo)  ================")
print(f"Numérica (solve_ivp moderado) vs Ref:  A: {err_num_A:.3e}   rho0: {err_num_rho:.3e}")
print(f"PINN vs Ref:                            A: {err_pinn_A:.3e}   rho0: {err_pinn_rho:.3e}")
print("====================================================================")
