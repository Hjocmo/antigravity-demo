import pandas as pd
import networkx as nx
import numpy as np
import time

# ==========================================
# CONFIGURATION MONTE CARLO
# ==========================================
N_SCENARIOS = 7500  # Nombre de combinaisons de paramètres (ex: alpha, theta, lambda)
N_REPS = 10         # Répétitions par scénario (W est recalculé à chaque fois)
RANDOM_SEED = 42    # Pour pouvoir reproduire les mêmes résultats
GAMMA = 1.0         # Contagion multiplier

# ==========================================
# PART 1: NETWORK GENERATION (FONCTION D'ORIGINE)
# ==========================================
def build_network(A, n, lambda_param):
    """
    Ta fonction originale, utilisée sans modification d'optimisation.
    Génère une nouvelle topologie et une nouvelle matrice W à chaque appel.
    """
    G_base = nx.barabasi_albert_graph(n, 3)
    G = G_base.to_directed()
    edges_to_remove = [e for e in G.edges() if np.random.rand() > 0.5]
    G.remove_edges_from(edges_to_remove)
    
    W = np.zeros((n, n))
    for i, j in G.edges():
        W[i, j] = A[i] * A[j]
        
    S = W.sum(axis=1)
    for i in range(n):
        if S[i] > 0:
            W[i, :] = lambda_param * A[i] * (W[i, :] / S[i])
            
    return W, G

# ==========================================
# MOTEUR PRINCIPAL MONTE CARLO
# ==========================================
def generate_ml_dataset():
    print(f"--- FENRIR: BOOTING MONTE CARLO ENGINE ---")
    print(f"Target: {N_SCENARIOS} scenarios x {N_REPS} reps = {N_SCENARIOS * N_REPS} propagation runs\n")
    
    np.random.seed(RANDOM_SEED)
    start_time = time.time()
    
    try:
        df = pd.read_csv("network_input.csv", on_bad_lines='warn')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Invariants du dataset
    A = df['TotalAssets'].values
    CR = df['CR'].values
    M = df['M_i_final'].values
    countries = df['Country'].values
    n_banks = len(df)
    
    C_init = CR * A
    total_assets = np.sum(A)
    unique_countries = np.unique(countries)
    
    # Invariants Macro
    mean_CR = np.mean(CR)
    mean_M = np.mean(M)
    share_high_M = np.mean(M > 0.7)
    assets_per_country = {c: np.sum(A[countries == c]) / total_assets for c in unique_countries}
    
    dataset = []

    # ==========================================
    # BOUCLE 1 : SCÉNARIOS
    # ==========================================
    for i in range(N_SCENARIOS):
        alpha = np.random.uniform(0.005, 0.02)
        theta = np.random.uniform(0.025, 0.05)
        lam = np.random.uniform(0.05, 0.25)
        shock_country = np.random.choice(unique_countries)
        
        share_assets_shock = assets_per_country[shock_country]
        shock_mask = (countries == shock_country).astype(float)
        
        metrics = {
            'net_density': 0, 'spectral_radius': 0, 'avg_degree': 0, 'clustering': 0,
            'init_def_ratio': 0, 'init_cap_loss': 0, 'mean_CR_after': 0,
            'L': [], 'N': [], 'depth': []
        }
        
        # ==========================================
        # BOUCLE 2 : RÉPÉTITIONS MONTE CARLO
        # ==========================================
        for rep in range(N_REPS):
            # 1. Le réseau est recalculé ici avec ton algorithme exact
            W, G = build_network(A, n_banks, lam)
            
            # Extraction des métriques du réseau
            metrics['net_density'] += nx.density(G)
            metrics['avg_degree'] += sum(dict(G.degree()).values()) / n_banks
            metrics['clustering'] += nx.average_clustering(G)
            eigenvalues = np.linalg.eigvals(W)
            metrics['spectral_radius'] += np.max(np.abs(eigenvalues))
            
            # 2. Choc Initial (sur le pays ciblé)
            C_0 = C_init - (alpha * shock_mask * M * A)
            D_0 = ((C_0 < (theta * A)) & (countries == shock_country)).astype(int)
            
            metrics['init_def_ratio'] += np.mean(D_0)
            metrics['init_cap_loss'] += np.sum(C_init - C_0)
            metrics['mean_CR_after'] += np.mean(C_0 / A)
            
            # 3. Moteur de Propagation
            C = C_0.copy()
            D = D_0.copy()
            t = 0
            while t < n_banks:
                C_new = C - GAMMA * W.dot(D)
                D_new = (C_new < (theta * A)).astype(int)
                
                if np.array_equal(D_new, D):
                    break
                    
                C = C_new
                D = D_new
                t += 1
                
            # 4. Calcul de la perte de la répétition
            loss_array = C_init - C
            total_loss = np.sum(loss_array[loss_array > 0])
            
            metrics['L'].append(total_loss)
            metrics['N'].append(np.sum(D))
            metrics['depth'].append(t)
            
        # ==========================================
        # AGGRÉGATION DE LA LIGNE POUR LE CSV
        # ==========================================
        row = {
            'alpha': alpha,
            'theta': theta,
            'lambda': lam,
            'shock_country': shock_country,
            
            'mean_CR': mean_CR,
            'mean_M': mean_M,
            'share_high_M': share_high_M,
            'share_assets_shock_country': share_assets_shock,
            
            'network_density': metrics['net_density'] / N_REPS,
            'spectral_radius': metrics['spectral_radius'] / N_REPS,
            'average_degree': metrics['avg_degree'] / N_REPS,
            'clustering_coefficient': metrics['clustering'] / N_REPS,
            
            'initial_defaults_ratio': metrics['init_def_ratio'] / N_REPS,
            'initial_total_capital_loss': metrics['init_cap_loss'] / N_REPS,
            'mean_CR_after_shock': metrics['mean_CR_after'] / N_REPS,
            
            'L_mean': np.mean(metrics['L']),
            'L_std': np.std(metrics['L']),
            'N_mean': np.mean(metrics['N']),
            'cascade_depth_mean': np.mean(metrics['depth'])
        }
        dataset.append(row)
        
        # Affichage de l'avancement
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Progress: {i + 1} / {N_SCENARIOS} scenarios aggregated.")
            
    # Export du Dataset Compact
    out_df = pd.DataFrame(dataset)
    out_df.to_csv("montecarlo_scenarios_compact.csv", index=False)
    print("\n✓ DATASET EXPORTED SUCCESSFULLY: 'montecarlo_scenarios_compact.csv'")
    print(f"Final shape: {out_df.shape} (Ready for ML training)")

if __name__ == "__main__":
    generate_ml_dataset()