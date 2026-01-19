#!/usr/bin/env python3
"""
test_rk6_order.py — Verify RK6 integrator order of convergence.

Tests the RK6 Butcher tableau on simple ODEs with known analytical solutions.
If the integrator is 6th order, the error should scale as O(dt^6).

Test problems:
1. Exponential decay: dy/dt = -y, y(0) = 1 → y(t) = e^(-t)
2. Harmonic oscillator: dy/dt = v, dv/dt = -y → y(t) = cos(t)
3. Nonlinear ODE: dy/dt = y^2, y(0) = 1 → y(t) = 1/(1-t)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rk6_step_generic(y, t, dt, rhs_func):
    """
    Generic RK6 Butcher step (7 stages, 6th order).

    Uses the same coefficients as in test_convergence_hydro.py.

    Args:
        y: current state (can be scalar or array)
        t: current time
        dt: time step
        rhs_func: function f(t, y) returning dy/dt

    Returns:
        y_new: state at t + dt
    """
    k1 = rhs_func(t, y)
    k2 = rhs_func(t + dt * (1/3), y + dt * (1/3) * k1)
    k3 = rhs_func(t + dt * (2/3), y + dt * (2/3) * k2)
    k4 = rhs_func(t + dt * (1/3), y + dt * (1/12 * k1 + 1/3 * k2 - 1/12 * k3))
    k5 = rhs_func(t + dt * (1/2), y + dt * (-1/16 * k1 + 9/8 * k2 - 3/16 * k3 - 3/8 * k4))
    k6 = rhs_func(t + dt * (1/2), y + dt * (9/8 * k2 - 3/8 * k3 - 3/4 * k4 + 1/2 * k5))
    k7 = rhs_func(t + dt, y + dt * (9/44 * k1 - 9/11 * k2 + 63/44 * k3 + 18/11 * k4 - 16/11 * k6))

    y_new = y + dt * (11/120 * k1 + 27/40 * k3 + 27/40 * k4 - 4/15 * k5 - 4/15 * k6 + 11/120 * k7)

    return y_new


def rk4_step_generic(y, t, dt, rhs_func):
    """
    Classic RK4 step (4th order) for comparison.
    """
    k1 = rhs_func(t, y)
    k2 = rhs_func(t + dt/2, y + dt/2 * k1)
    k3 = rhs_func(t + dt/2, y + dt/2 * k2)
    k4 = rhs_func(t + dt, y + dt * k3)

    y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_new


def integrate(y0, t_final, dt, rhs_func, stepper):
    """Integrate ODE from t=0 to t_final using given stepper."""
    t = 0.0
    y = np.copy(y0)

    while t < t_final - 1e-14:
        if t + dt > t_final:
            dt = t_final - t
        y = stepper(y, t, dt, rhs_func)
        t += dt

    return y


def test_exponential_decay():
    """Test: dy/dt = -y, y(0) = 1 → y(t) = exp(-t)"""
    print("\n" + "="*70)
    print("TEST 1: Exponential Decay  dy/dt = -y, y(0) = 1")
    print("        Exact solution: y(t) = exp(-t)")
    print("="*70)

    def rhs(t, y):
        return -y

    def exact(t):
        return np.exp(-t)

    y0 = 1.0
    t_final = 1.0

    # Test with different time steps
    dts = [0.2, 0.1, 0.05, 0.025, 0.0125]

    errors_rk6 = []
    errors_rk4 = []

    print(f"\n{'dt':<12} {'Error RK6':<14} {'Error RK4':<14} {'Ratio RK6':<12} {'Ratio RK4':<12}")
    print("-"*70)

    for i, dt in enumerate(dts):
        y_rk6 = integrate(y0, t_final, dt, rhs, rk6_step_generic)
        y_rk4 = integrate(y0, t_final, dt, rhs, rk4_step_generic)
        y_exact = exact(t_final)

        err_rk6 = abs(y_rk6 - y_exact)
        err_rk4 = abs(y_rk4 - y_exact)

        errors_rk6.append(err_rk6)
        errors_rk4.append(err_rk4)

        ratio_rk6 = errors_rk6[i-1] / err_rk6 if i > 0 and err_rk6 > 0 else 0
        ratio_rk4 = errors_rk4[i-1] / err_rk4 if i > 0 and err_rk4 > 0 else 0

        print(f"{dt:<12.4f} {err_rk6:<14.2e} {err_rk4:<14.2e} {ratio_rk6:<12.1f} {ratio_rk4:<12.1f}")

    # Compute convergence orders
    orders_rk6 = []
    orders_rk4 = []
    for i in range(1, len(dts)):
        if errors_rk6[i] > 0 and errors_rk6[i-1] > 0:
            order = np.log(errors_rk6[i-1] / errors_rk6[i]) / np.log(dts[i-1] / dts[i])
            orders_rk6.append(order)
        if errors_rk4[i] > 0 and errors_rk4[i-1] > 0:
            order = np.log(errors_rk4[i-1] / errors_rk4[i]) / np.log(dts[i-1] / dts[i])
            orders_rk4.append(order)

    avg_order_rk6 = np.mean(orders_rk6) if orders_rk6 else 0
    avg_order_rk4 = np.mean(orders_rk4) if orders_rk4 else 0

    print(f"\nConvergence orders: RK6 = {orders_rk6}")
    print(f"                    RK4 = {orders_rk4}")
    print(f"Average order: RK6 = {avg_order_rk6:.2f} (expected 6)")
    print(f"               RK4 = {avg_order_rk4:.2f} (expected 4)")

    # For 6th order: error ratio should be ~2^6 = 64 when dt halves
    print(f"\nExpected ratio when dt halves: RK6 → 2^6 = 64, RK4 → 2^4 = 16")

    return avg_order_rk6, errors_rk6, dts


def test_harmonic_oscillator():
    """Test: dy/dt = v, dv/dt = -y → y(t) = cos(t), v(t) = -sin(t)"""
    print("\n" + "="*70)
    print("TEST 2: Harmonic Oscillator  d²y/dt² = -y")
    print("        Exact: y(t) = cos(t), v(t) = -sin(t)")
    print("="*70)

    def rhs(t, state):
        y, v = state
        return np.array([v, -y])

    def exact(t):
        return np.array([np.cos(t), -np.sin(t)])

    y0 = np.array([1.0, 0.0])  # y(0) = 1, v(0) = 0
    t_final = 2 * np.pi  # One full period

    dts = [0.4, 0.2, 0.1, 0.05, 0.025]

    errors_rk6 = []
    errors_rk4 = []

    print(f"\n{'dt':<12} {'Error RK6':<14} {'Error RK4':<14} {'Order RK6':<12} {'Order RK4':<12}")
    print("-"*70)

    for i, dt in enumerate(dts):
        y_rk6 = integrate(y0, t_final, dt, rhs, rk6_step_generic)
        y_rk4 = integrate(y0, t_final, dt, rhs, rk4_step_generic)
        y_exact = exact(t_final)

        err_rk6 = np.linalg.norm(y_rk6 - y_exact)
        err_rk4 = np.linalg.norm(y_rk4 - y_exact)

        errors_rk6.append(err_rk6)
        errors_rk4.append(err_rk4)

        if i > 0 and err_rk6 > 0 and errors_rk6[i-1] > 0:
            order_rk6 = np.log(errors_rk6[i-1] / err_rk6) / np.log(dts[i-1] / dt)
        else:
            order_rk6 = 0

        if i > 0 and err_rk4 > 0 and errors_rk4[i-1] > 0:
            order_rk4 = np.log(errors_rk4[i-1] / err_rk4) / np.log(dts[i-1] / dt)
        else:
            order_rk4 = 0

        print(f"{dt:<12.4f} {err_rk6:<14.2e} {err_rk4:<14.2e} {order_rk6:<12.2f} {order_rk4:<12.2f}")

    # Compute average order
    orders_rk6 = []
    orders_rk4 = []
    for i in range(1, len(dts)):
        if errors_rk6[i] > 0 and errors_rk6[i-1] > 0:
            orders_rk6.append(np.log(errors_rk6[i-1] / errors_rk6[i]) / np.log(dts[i-1] / dts[i]))
        if errors_rk4[i] > 0 and errors_rk4[i-1] > 0:
            orders_rk4.append(np.log(errors_rk4[i-1] / errors_rk4[i]) / np.log(dts[i-1] / dts[i]))

    avg_order_rk6 = np.mean(orders_rk6) if orders_rk6 else 0
    avg_order_rk4 = np.mean(orders_rk4) if orders_rk4 else 0

    print(f"\nAverage order: RK6 = {avg_order_rk6:.2f} (expected 6)")
    print(f"               RK4 = {avg_order_rk4:.2f} (expected 4)")

    return avg_order_rk6, errors_rk6, dts


def test_nonlinear_ode():
    """Test: dy/dt = y², y(0) = 1 → y(t) = 1/(1-t)"""
    print("\n" + "="*70)
    print("TEST 3: Nonlinear ODE  dy/dt = y², y(0) = 1")
    print("        Exact: y(t) = 1/(1-t) (blows up at t=1)")
    print("="*70)

    def rhs(t, y):
        return y * y

    def exact(t):
        return 1.0 / (1.0 - t)

    y0 = 1.0
    t_final = 0.5  # Stay away from singularity at t=1

    dts = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    errors_rk6 = []
    errors_rk4 = []

    print(f"\n{'dt':<12} {'Error RK6':<14} {'Error RK4':<14} {'Order RK6':<12} {'Order RK4':<12}")
    print("-"*70)

    for i, dt in enumerate(dts):
        y_rk6 = integrate(y0, t_final, dt, rhs, rk6_step_generic)
        y_rk4 = integrate(y0, t_final, dt, rhs, rk4_step_generic)
        y_exact = exact(t_final)

        err_rk6 = abs(y_rk6 - y_exact)
        err_rk4 = abs(y_rk4 - y_exact)

        errors_rk6.append(err_rk6)
        errors_rk4.append(err_rk4)

        if i > 0 and err_rk6 > 0 and errors_rk6[i-1] > 0:
            order_rk6 = np.log(errors_rk6[i-1] / err_rk6) / np.log(dts[i-1] / dt)
        else:
            order_rk6 = 0

        if i > 0 and err_rk4 > 0 and errors_rk4[i-1] > 0:
            order_rk4 = np.log(errors_rk4[i-1] / err_rk4) / np.log(dts[i-1] / dt)
        else:
            order_rk4 = 0

        print(f"{dt:<12.5f} {err_rk6:<14.2e} {err_rk4:<14.2e} {order_rk6:<12.2f} {order_rk4:<12.2f}")

    # Compute average order
    orders_rk6 = []
    orders_rk4 = []
    for i in range(1, len(dts)):
        if errors_rk6[i] > 0 and errors_rk6[i-1] > 0:
            orders_rk6.append(np.log(errors_rk6[i-1] / errors_rk6[i]) / np.log(dts[i-1] / dts[i]))
        if errors_rk4[i] > 0 and errors_rk4[i-1] > 0:
            orders_rk4.append(np.log(errors_rk4[i-1] / errors_rk4[i]) / np.log(dts[i-1] / dts[i]))

    avg_order_rk6 = np.mean(orders_rk6) if orders_rk6 else 0
    avg_order_rk4 = np.mean(orders_rk4) if orders_rk4 else 0

    print(f"\nAverage order: RK6 = {avg_order_rk6:.2f} (expected 6)")
    print(f"               RK4 = {avg_order_rk4:.2f} (expected 4)")

    return avg_order_rk6, errors_rk6, dts


def test_stiff_linear_system():
    """Test on a stiff linear system to see stability."""
    print("\n" + "="*70)
    print("TEST 4: Stiff Linear System  dy/dt = Ay")
    print("        A = [[-1, 0], [0, -100]] (stiffness ratio = 100)")
    print("="*70)

    # Stiff system: eigenvalues -1 and -100
    A = np.array([[-1.0, 0.0], [0.0, -100.0]])

    def rhs(t, y):
        return A @ y

    def exact(t, y0):
        return np.array([y0[0] * np.exp(-t), y0[1] * np.exp(-100*t)])

    y0 = np.array([1.0, 1.0])
    t_final = 0.1  # Short time due to stiffness

    dts = [0.02, 0.01, 0.005, 0.0025, 0.00125]

    errors_rk6 = []

    print(f"\n{'dt':<12} {'Error RK6':<14} {'Order RK6':<12}")
    print("-"*50)

    for i, dt in enumerate(dts):
        y_rk6 = integrate(y0, t_final, dt, rhs, rk6_step_generic)
        y_exact = exact(t_final, y0)

        err_rk6 = np.linalg.norm(y_rk6 - y_exact)
        errors_rk6.append(err_rk6)

        if i > 0 and err_rk6 > 0 and errors_rk6[i-1] > 0:
            order_rk6 = np.log(errors_rk6[i-1] / err_rk6) / np.log(dts[i-1] / dt)
        else:
            order_rk6 = 0

        print(f"{dt:<12.5f} {err_rk6:<14.2e} {order_rk6:<12.2f}")

    orders_rk6 = []
    for i in range(1, len(dts)):
        if errors_rk6[i] > 0 and errors_rk6[i-1] > 0:
            orders_rk6.append(np.log(errors_rk6[i-1] / errors_rk6[i]) / np.log(dts[i-1] / dts[i]))

    avg_order = np.mean(orders_rk6) if orders_rk6 else 0
    print(f"\nAverage order: RK6 = {avg_order:.2f} (expected 6)")

    return avg_order


def plot_convergence(results):
    """Plot convergence results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    test_names = ["Exponential Decay", "Harmonic Oscillator", "Nonlinear ODE"]

    for i, (order, errors, dts) in enumerate(results):
        ax = axes[i]

        # Plot errors
        ax.loglog(dts, errors, 'bo-', linewidth=2, markersize=8, label=f'RK6 (order={order:.2f})')

        # Reference slopes
        dt_ref = np.array(dts)
        err_scale = errors[0] * (dt_ref / dts[0])**6
        ax.loglog(dt_ref, err_scale, 'k--', alpha=0.5, label='6th order slope')

        err_scale_4 = errors[0] * (dt_ref / dts[0])**4
        ax.loglog(dt_ref, err_scale_4, 'k:', alpha=0.5, label='4th order slope')

        ax.set_xlabel('dt')
        ax.set_ylabel('Error')
        ax.set_title(test_names[i])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_rk6_order.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: test_rk6_order.png")


def main():
    print("="*70)
    print("RK6 INTEGRATOR ORDER VERIFICATION")
    print("="*70)
    print("\nThis test verifies that the RK6 Butcher tableau achieves 6th order")
    print("convergence on various ODEs with known analytical solutions.")
    print("\nFor an order-p method, error ~ O(dt^p), so when dt halves,")
    print("the error should decrease by a factor of 2^p.")
    print("  - RK4: error ratio = 2^4 = 16 when dt halves")
    print("  - RK6: error ratio = 2^6 = 64 when dt halves")

    results = []

    # Run all tests
    order1, errors1, dts1 = test_exponential_decay()
    results.append((order1, errors1, dts1))

    order2, errors2, dts2 = test_harmonic_oscillator()
    results.append((order2, errors2, dts2))

    order3, errors3, dts3 = test_nonlinear_ode()
    results.append((order3, errors3, dts3))

    order4 = test_stiff_linear_system()

    # Plot results
    plot_convergence(results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_orders = [order1, order2, order3, order4]
    avg_order = np.mean(all_orders)

    print(f"\nMeasured orders: {[f'{o:.2f}' for o in all_orders]}")
    print(f"Average order: {avg_order:.2f}")
    print(f"Expected order: 6.00")

    # Pass/fail criteria: average order should be > 5.5
    if avg_order > 5.5:
        print("\nRESULT: PASS - RK6 integrator achieves 6th order convergence")
        return True
    else:
        print(f"\nRESULT: FAIL - Order {avg_order:.2f} is below expected 6")
        return False


if __name__ == "__main__":
    ok = main()
    print("\n" + "="*70)
    print("FINAL:", "PASS" if ok else "FAIL")
    print("="*70)
