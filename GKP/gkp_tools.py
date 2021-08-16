import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


def wigner(rho, xvec, yvec=None):
    if yvec is None:
        yvec = xvec
    return (np.pi / 2.0) * qt.wigner(rho, xvec, yvec, g=2)


def plot_wigner(psi, xvec=np.linspace(-5, 5, 41), ax=None, grid=True):
    W = wigner(psi, xvec)
    plot_wigner_data(W, xvec, ax=ax, grid=grid)


def plot_wigner_data(W, xvec=np.linspace(-5, 5, 41), ax=None, grid=True, yvec=None):
    yvec = xvec if yvec is None else yvec
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    extent = (
        xvec[0] - dx / 2.0,
        xvec[-1] + dx / 2.0,
        yvec[0] - dy / 2.0,
        yvec[-1] + dy / 2.0,
    )
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(
        W,
        origin="lower",
        extent=extent,
        cmap="seismic",
        vmin=-1,
        vmax=+1,
        interpolation=None,
    )
    # plt.colorbar()
    if grid:
        ax.grid()


def plot_results(results):
    ts_us = np.array(results["ts"]) / 1e3
    steps = results["steps"]
    fig, axs = plt.subplots(1, 3, sharey=False, figsize=(11, 4))
    axs[0].plot(ts_us, np.real(results["Sp"]), ".", label="Re(<Sp>)")
    axs[0].plot(ts_us, np.real(results["Sq"]), ".", label="Re(<Sq>)")
    axs[0].plot(ts_us, np.imag(results["Sp"]), ".", label="Im(<Sp>)")
    axs[0].plot(ts_us, np.imag(results["Sq"]), ".", label="Im(<Sq>)")
    axs[0].legend()
    axs[0].set_xlabel("t μs")
    ax2 = axs[0].twiny()
    ax2.set_xlabel("steps")
    ax2.set_xlim(steps[0], steps[-1])
    ax2.set_xticks(np.linspace(steps[0], steps[-1], 5).astype(int))
    # ax2.set_xticklabels(['7','8','99'])
    # ax2.cla()
    axs[1].plot(ts_us, np.real(results["X"]), ".", label="Re(<X>)")
    axs[1].plot(ts_us, np.real(results["Y"]), ".", label="Re(<Y>)")
    axs[1].plot(ts_us, np.real(results["Z"]), ".", label="Re(<Z>)")
    axs[1].set_xlabel("t μs")
    ax2 = axs[1].twiny()
    ax2.set_xlabel("steps")
    ax2.set_xlim(steps[0], steps[-1])
    ax2.set_xticks(np.linspace(steps[0], steps[-1], 5).astype(int))
    # axs[1].plot(np.imag(results['X']),'.:', label='Im(<X>)')
    # axs[1].plot(np.imag(results['Y']),'.:', label='Im(<Y>)')
    # axs[1].plot(np.imag(results['X']),'.:', label='Im(<Z>)')
    axs[1].legend()
    axs[2].plot(ts_us, results["q"], ".:", label="<q>")
    axs[2].plot(ts_us, np.real(results["p"]), ".", label="<p>")
    axs[2].plot(ts_us, np.real(results["n"]), ".", label="<n>")
    axs[2].legend()
    axs[2].set_xlabel("t μs")
    ax2 = axs[2].twiny()
    ax2.set_xlabel("steps")
    ax2.set_xlim(steps[0], steps[-1])
    ax2.set_xticks(np.linspace(steps[0], steps[-1], 5).astype(int))


def save_results(results, filename):
    np.savez(filename, **results)


def load_results(filename):
    f = np.load(filename, allow_pickle=True)
    r = {}
    for k in f.files:
        print(k)
        r[k] = f[k]
    return r
