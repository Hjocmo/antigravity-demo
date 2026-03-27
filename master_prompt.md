# CONTEXT AND OBJECTIVE
You are a Senior Full-Stack Developer and Data Scientist. 
We are building a prototype for a systemic banking risk simulation app (contagion model). 
This is a **Beta UI Demo** meant to prove the feasibility of the concept to stakeholders. Do NOT build a perfect mathematical optimization model or use Machine Learning. Prioritize clean, modular code, speed, and a highly convincing visual effect.

# REFERENCE FILES PROVIDED
- `fenrir.py`: This is our heavy Monte Carlo dataset generator. DO NOT run this file. Instead, extract the core logic from `build_network` and the contagion loop (`C_new = C - GAMMA * W.dot(D)`) to use in our new, simplified engine.
- `network_input.csv`: Use this structure for the bank nodes.

# REQUIRED TECH STACK
- Backend: Python (Pandas, NumPy, NetworkX)
- Frontend/UI: Streamlit 
- Visualization: Plotly or PyVis for an interactive network graph (Streamlit-compatible).

# FILE ARCHITECTURE TO CREATE
Structure the project exactly like this:
1. `app.py`: Streamlit UI.
2. `engine.py`: Deterministic propagation engine.
3. `intervention.py`: Heuristic recapitalization algorithm.
4. `utils.py`: Mock data generation and graph drawing helpers.

# TECHNICAL BLUEPRINT

## 1. Propagation Engine (engine.py)
Create `simulate(alpha, theta, lambda_, shock_country, df_nodes)`
- Run ONE deterministic trajectory (No Monte Carlo loops).
- Shock the selected country: `C_0 = C_init - (alpha * shock_mask * M * A)`
- Run the while loop until no new defaults occur.
- **Return:** A dictionary with `losses_per_wave`, `defaults_per_wave`, `final_total_losses`, and `final_network_state`.

## 2. Heuristic Optimization Module (intervention.py)
Create `heuristic_recapitalization(budget_B, initial_state, shock_country, alpha, theta, lambda_)`
- **Logic:** 1. Simulate the initial shock (wave 0) to see which banks are about to default.
  2. Rank the vulnerable banks by their systemic importance (e.g., Highest TotalAssets or Highest Out-Degree).
  3. Loop through this sorted list: inject just enough capital into each bank so its Capital Ratio (CR) goes slightly above `theta`. Deduct this from `budget_B`.
  4. Stop when `budget_B` is empty.
  5. Run the `simulate()` function again with this new injected capital to see the improved result.
- **Return:** `banks_saved`, `capital_spent`, `new_final_losses`.

## 3. User Interface (app.py)
Follow this flow:
- **Sidebar:** Sliders for Alpha, Theta, Lambda, Budget. Dropdown for Country. "Run Simulation" button.
- **Top Row (Before):** Display KPI columns (Defaults, Total Losses). Show the network graph (defaulted nodes in red).
- **Bottom Row (After Intervention):** Display the recommendation ("Injected X into Y banks"). Show the new KPIs (Losses saved). Show the stabilized network graph (saved nodes in green).

# FIRST STEP
Generate ONLY the file structure and the empty functions with their docstrings (the skeleton). Wait for my validation before writing the mathematical logic inside them.