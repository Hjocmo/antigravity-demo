"""
intervention.py — Heuristic Recapitalization Algorithm
=======================================================
Given a finite public budget B, determines the optimal allocation of
emergency capital injections to minimise systemic losses.
"""

import numpy as np
import pandas as pd
from engine import simulate


def heuristic_recapitalization(
    budget_B: float,
    df_nodes: pd.DataFrame,
    shock_country: str,
    alpha: float,
    theta: float,
    lambda_: float,
    seed: int = 42,
) -> dict:
    """
    Heuristic greedy recapitalization policy.

    Logic
    -----
    1. Run `simulate()` to observe which banks are about to default after
       the initial shock (wave 0).
    2. Rank the vulnerable banks by systemic importance
       (TotalAssets descending, then out-degree descending as tiebreaker).
    3. Iterate through the ranked list: inject the minimum capital needed
       to bring each bank's Capital Ratio (CR) just above `theta`.
       Deduct the injected amount from `budget_B`.
    4. Stop when `budget_B` is exhausted or all vulnerable banks are saved.
    5. Run `simulate()` again with the augmented capital DataFrame to
       measure the improvement.

    Parameters
    ----------
    budget_B : float
        Total public rescue budget available (EUR millions).
    df_nodes : pd.DataFrame
        Node-level data (same format as network_input.csv).
    shock_country : str
        Country to which the initial shock is applied.
    alpha : float
        Shock severity parameter.
    theta : float
        Minimum Capital Ratio threshold for default.
    lambda_ : float
        Contagion intensity parameter.

    Returns
    -------
    dict with keys:
        - 'banks_saved'      : int   — number of banks rescued.
        - 'capital_spent'    : float — total capital injected (EUR millions).
        - 'new_final_losses' : float — aggregate losses after intervention.
        - 'df_intervened'    : pd.DataFrame — df_nodes with updated CR /
                               capital reflecting injections.
        - 'sim_after'        : dict  — full result dict from the post-intervention
                               simulate() call.
    """
    # ------------------------------------------------------------------
    # 1.  Run the BASELINE simulation to identify wave-0 vulnerable banks.
    #     We look at who defaults *immediately after* the initial shock —
    #     these are the targets for recapitalization before contagion spreads.
    # ------------------------------------------------------------------
    # Both simulate() calls must use the same seed so they produce identical
    # network topologies — this is what makes before/after counts comparable.
    sim_before = simulate(alpha, theta, lambda_, shock_country, df_nodes, seed=seed)

    # Grab the network graph so we can rank by out-degree (systemic importance)
    G_before = sim_before['G']

    # Extract the post-shock capital vector (wave 0 only, before contagion)
    A        = df_nodes['TotalAssets'].values.astype(float)
    CR       = df_nodes['CR'].values.astype(float)
    M        = df_nodes['M_i_final'].values.astype(float)
    countries = df_nodes['Country'].values

    C_init   = CR * A
    shock_mask = (countries == shock_country).astype(float)
    C_0      = C_init - (alpha * shock_mask * M * A)

    # Wave-0 default indicator — mirrors engine.py D_0 logic exactly
    D_0 = ((C_0 < (theta * A)) & (countries == shock_country)).astype(int)

    # Indices of banks that defaulted at wave 0 (the rescue candidates)
    vulnerable_idx = np.where(D_0 == 1)[0]

    # ------------------------------------------------------------------
    # 2.  Rank vulnerable banks by systemic importance.
    #
    #     Primary key   : TotalAssets (descending) — bigger banks cause
    #                     larger contagion losses when they fail.
    #     Secondary key : out-degree in G (descending) — more connected
    #                     banks spread contagion to more counterparties.
    # ------------------------------------------------------------------
    out_degrees = dict(G_before.out_degree())  # {node_id: out_degree}

    def sort_key(idx):
        return (
            -df_nodes.iloc[idx]['TotalAssets'],   # primary: largest first
            -out_degrees.get(idx, 0)               # secondary: most connected first
        )

    ranked_idx = sorted(vulnerable_idx, key=sort_key)

    # ------------------------------------------------------------------
    # 3.  Greedy injection loop.
    #
    #     For each ranked bank, compute the minimum injection needed to
    #     raise its post-shock capital ratio just above theta:
    #
    #       target_C   = theta * A[i]  (the regulatory floor)
    #       injection  = max(0, target_C - C_0[i]) + epsilon
    #
    #     epsilon = 1e-6 ensures the bank is strictly above, not exactly
    #     at, the threshold (so it won't re-default on the next D_new check).
    #
    #     Inject only if the remaining budget covers the cost.
    # ------------------------------------------------------------------
    epsilon          = 1e-6          # tiny buffer above the threshold
    df_intervened    = df_nodes.copy()
    total_spent      = 0.0
    banks_saved_list = []            # track which banks were actually saved
    remaining_budget = float(budget_B)

    for idx in ranked_idx:
        bank_A     = A[idx]
        target_C   = theta * bank_A    # capital floor (EUR millions)
        shortfall  = target_C - C_0[idx] + epsilon   # how much to inject

        if shortfall <= 0:
            # Bank is already above threshold — no injection needed
            continue

        if shortfall > remaining_budget:
            # Can't afford this bank; skip to the next (greedy, not fractional)
            continue

        # Inject: update the CR in the working DataFrame so simulate() picks
        # it up. The dataframe's 'CR' is pre-shock capital, so we must add
        # the shortfall to C_init[idx], not C_0[idx] (otherwise simulate()
        # will subtract the shock twice!).
        # New CR = (C_init[idx] + shortfall) / A[idx]
        new_cap_ratio = (C_init[idx] + shortfall) / bank_A
        df_intervened.at[df_intervened.index[idx], 'CR'] = new_cap_ratio

        remaining_budget -= shortfall
        total_spent      += shortfall
        banks_saved_list.append(idx)

    # ------------------------------------------------------------------
    # 4.  Re-run the simulation on the augmented DataFrame.
    #     The rescued banks now have higher CRs so they will not default
    #     at wave 0, breaking the contagion cascade.
    # ------------------------------------------------------------------
    sim_after = simulate(alpha, theta, lambda_, shock_country, df_intervened, seed=seed)

    # ------------------------------------------------------------------
    # 5.  Mark rescued banks in the post-intervention network state so
    #     utils.build_plotly_network() can colour them green.
    # ------------------------------------------------------------------
    sim_after['final_network_state']['rescued'] = False
    for idx in banks_saved_list:
        row_label = df_intervened.index[idx]
        sim_after['final_network_state'].at[row_label, 'rescued'] = True

    return {
        'banks_saved'      : len(banks_saved_list),
        'capital_spent'    : total_spent,
        'new_final_losses' : sim_after['final_total_losses'],
        'df_intervened'    : df_intervened,
        'sim_after'        : sim_after,
    }
