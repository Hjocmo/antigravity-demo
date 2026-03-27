"""
engine.py — Deterministic Contagion Propagation Engine
=======================================================
Extracted and simplified from fenrir.py.
Runs a single deterministic trajectory (no Monte Carlo loops).
"""

import numpy as np
import pandas as pd
import networkx as nx

# Contagion multiplier (from fenrir.py — must never be changed here)
GAMMA = 1.0


def build_network(A: np.ndarray, n: int, lambda_: float, seed: int = 42):
    """
    Build a directed interbank exposure network using a Barabási–Albert
    scale-free topology, then calibrate edge weights based on total assets.

    Adapted directly from fenrir.py `build_network()`.

    Parameters
    ----------
    A : np.ndarray
        Array of total assets for each bank (shape: [n]).
    n : int
        Number of banks (nodes).
    lambda_ : float
        Contagion intensity parameter controlling the row-sum of W.
    seed : int
        Random seed for reproducibility.  Both the BA graph and the edge-pruning
        step are seeded so that every call with the same arguments produces an
        identical W and G.  This is critical for making the before/after
        intervention results directly comparable.

    Returns
    -------
    W : np.ndarray
        Calibrated n×n exposure/weight matrix.
    G : nx.DiGraph
        The directed interbank network graph.
    """
    # Step 1 — Fix the RNG seed so this function is fully deterministic.
    #           Without this, each call generates a different random graph,
    #           making before/after comparisons meaningless.
    np.random.seed(seed)

    # Step 2 — Generate Barabási–Albert undirected base graph, then make directed.
    #           This gives us the scale-free topology (hub-and-spoke structure).
    G_base = nx.barabasi_albert_graph(n, 3, seed=seed)
    G = G_base.to_directed()

    # Step 3 — Randomly prune ~50 % of directed edges (as in fenrir.py).
    #           np.random is already seeded above, so pruning is reproducible.
    edges_to_remove = [e for e in G.edges() if np.random.rand() > 0.5]
    G.remove_edges_from(edges_to_remove)

    # Step 4 — Build raw weight matrix W where W[i,j] = A[i] * A[j].
    #           Larger banks create larger bilateral exposures.
    W = np.zeros((n, n))
    for i, j in G.edges():
        W[i, j] = A[i] * A[j]

    # Step 5 — Normalise each row so that the row-sum equals lambda_ * A[i].
    #           This means bank i transmits at most lambda_ times its own assets
    #           to its counterparties in total — a meaningful economic bound.
    S = W.sum(axis=1)          # total raw exposure per bank (row-sum)
    for i in range(n):
        if S[i] > 0:
            W[i, :] = lambda_ * A[i] * (W[i, :] / S[i])

    return W, G


def simulate(alpha: float, theta: float, lambda_: float,
             shock_country: str, df_nodes: pd.DataFrame,
             seed: int = 42) -> dict:
    """
    Run ONE deterministic contagion trajectory from an initial country shock.

    Parameters
    ----------
    alpha : float
        Shock severity parameter (fraction of macro-risk-adjusted capital lost).
    theta : float
        Minimum Capital Ratio threshold; banks below this are considered in default.
    lambda_ : float
        Contagion intensity parameter for network weight calibration.
    shock_country : str
        The country to which the initial shock is applied.
    df_nodes : pd.DataFrame
        Node-level data loaded from network_input.csv.
        Required columns: TotalAssets, CR, M_i_final, Country, BankName.

    Returns
    -------
    dict with keys:
        - 'losses_per_wave'     : list[float] — total capital loss at each wave.
        - 'defaults_per_wave'   : list[int]   — number of defaulted banks per wave.
        - 'final_total_losses'  : float       — aggregate capital loss at convergence.
        - 'final_network_state' : pd.DataFrame — df_nodes enriched with final
                                   columns: C_final (float), defaulted (bool).
        - 'W'                   : np.ndarray  — the exposure matrix used.
        - 'G'                   : nx.DiGraph  — the interbank network graph.
    """
    # ------------------------------------------------------------------
    # 0.  Extract raw data arrays from df_nodes
    # ------------------------------------------------------------------
    A        = df_nodes['TotalAssets'].values.astype(float)   # total assets
    CR       = df_nodes['CR'].values.astype(float)            # capital ratios
    M        = df_nodes['M_i_final'].values.astype(float)     # macro-risk index
    countries = df_nodes['Country'].values                    # ISO country codes
    n        = len(df_nodes)

    # ------------------------------------------------------------------
    # 1.  Derive initial capital buffers
    #     C_init[i] = CR[i] * A[i]  — monetary capital held by bank i
    # ------------------------------------------------------------------
    C_init = CR * A

    # ------------------------------------------------------------------
    # 2.  Build the interbank exposure network (single deterministic call)
    # ------------------------------------------------------------------
    W, G = build_network(A, n, lambda_, seed=seed)

    # ------------------------------------------------------------------
    # 3.  Apply the initial country-level shock
    #
    #     Only banks in shock_country are hit.
    #     shock_mask[i] = 1 if bank i is in the shocked country, else 0.
    #
    #     C_0[i] = C_init[i] - alpha * shock_mask[i] * M[i] * A[i]
    #
    #     Interpretation: bank i loses a fraction alpha of (M[i] * A[i])
    #     — its "risk-adjusted asset base" — due to the macro shock.
    # ------------------------------------------------------------------
    shock_mask = (countries == shock_country).astype(float)
    C_0 = C_init - (alpha * shock_mask * M * A)

    # ------------------------------------------------------------------
    # 4.  Identify defaults after the initial shock
    #     Bank i defaults if its capital buffer falls below theta * A[i].
    #     D_0[i] = 1 if C_0[i] < theta * A[i], else 0
    #     NOTE: in fenrir.py wave-0 defaults are also conditioned on the
    #     shock country, so we match that exactly.
    # ------------------------------------------------------------------
    D_0 = ((C_0 < (theta * A)) & (countries == shock_country)).astype(int)

    # ------------------------------------------------------------------
    # 5.  Record wave 0 metrics before contagion spreads
    # ------------------------------------------------------------------
    wave_losses    = [float(np.sum(np.maximum(C_init - C_0, 0)))]
    wave_defaults  = [int(np.sum(D_0))]

    # ------------------------------------------------------------------
    # 6.  Contagion propagation loop — identical to fenrir.py's while-loop.
    #
    #     At each wave t:
    #       C_new[i] = C[i] - GAMMA * sum_j( W[i,j] * D[j] )
    #
    #     Interpretation: defaulted banks (D[j] = 1) impose mark-to-market
    #     losses on their creditors via the weighted exposure matrix W.
    #     GAMMA = 1.0 means the full exposure amount is lost.
    #
    #     Convergence: the loop stops when no new bank crosses the default
    #     threshold (D_new == D), or when the cascade depth reaches n_banks
    #     (safety cap identical to fenrir.py).
    # ------------------------------------------------------------------
    C = C_0.copy()
    D = D_0.copy()
    t = 0

    while t < n:
        # Contagion update — core equation from fenrir.py
        C_new = C - GAMMA * W.dot(D)

        # Any bank whose capital now falls below the regulatory threshold defaults
        D_new = (C_new < (theta * A)).astype(int)

        # Convergence check
        if np.array_equal(D_new, D):
            break

        # Advance state
        C = C_new
        D = D_new
        t += 1

        # Track per-wave metrics (loss vs. C_init, not previous wave)
        loss_this_wave = float(np.sum(np.maximum(C_init - C, 0)))
        wave_losses.append(loss_this_wave)
        wave_defaults.append(int(np.sum(D)))

    # ------------------------------------------------------------------
    # 7.  Compile final network state DataFrame
    # ------------------------------------------------------------------
    df_result = df_nodes.copy()
    df_result['C_final']  = C
    df_result['defaulted'] = D.astype(bool)

    # Final aggregate capital loss across all banks that lost capital
    loss_array = C_init - C
    final_total_losses = float(np.sum(loss_array[loss_array > 0]))

    return {
        'losses_per_wave'     : wave_losses,
        'defaults_per_wave'   : wave_defaults,
        'final_total_losses'  : final_total_losses,
        'final_network_state' : df_result,
        'W'                   : W,
        'G'                   : G,
    }
