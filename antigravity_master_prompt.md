# Antigravity | Master Project Specification

You are an expert full-stack developer and UI/UX engineer building a premium, dark-themed SaaS application focusing on systemic banking risk. This tool models financial contagion across a European interbank network.

## 1. Technical Stack
* **Python 3.10+** (Backend logic and data processing)
* **Streamlit (`streamlit`)** (Frontend framework containing heavy Custom CSS overrides)
* **NetworkX (`networkx`)** (Deterministic scale-free graph modeling)
* **Pandas (`pandas`)** (Data manipulation)
* **Plotly (`plotly.graph_objects`)** (Geographic 3D Orthographic networks)
* **Geopy (`geopy`)** (Address-to-coordinate conversion)

## 2. Global Architecture & File Structure
The project strictly consists of 4 core modules:
1. `utils.py`: Geographic mapping (Nominatim geocoding) with deterministic micro-jittering to prevent node overlap. It handles the 3D Plotly Map rendering.
2. `engine.py`: Defines the systemic risk model (contagion propagation using a deterministic Barabási–Albert topology).
3. `intervention.py`: Implements a greedy recapitalization algorithm (heuristic policy) taking public rescue budgets to save vulnerable nodes.
4. `app.py`: The main Streamlit dashboard injecting CSS layouts, parsing states, and rendering the 3-column UI.

## 3. Engine Logic (`engine.py`)
* Use `np.random.seed(seed)` globally to ensure reproducibility in `build_network()`.
* **Topology:** Generate a directed scale-free network using `nx.barabasi_albert_graph(n, 3)`. Apply uniform random pruning (50% removal) to limit density. 
* **Weights:** `W[i,j] = lambda_ * A[i] * (A[i]*A[j] / Sum(k in N(i): A[i]*A[k]))` mapping exposure proportional to total assets `A`.
* **Simulation:** The first wave shocks a specific country (nodes lose `alpha` fraction of their risk-adjusted capital). If Capital Ratio (CR) drops below `theta`, the bank defaults and passes its interbank exposures as losses to its creditors in subsequent waves.
* **Return Formats:** Ensure `simulate()` returns a dict featuring `'final_network_state': df` (which MUST contain the boolean column `'defaulted'` and final capital `'C_final'`).

## 4. Intervention Logic (`intervention.py`)
* Implement a `heuristic_recapitalization()` function.
* **Algorithm:**
    1. Run `simulate()` to find banks defaulting in wave 0.
    2. Rank these vulnerable banks by Total Assets (descending).
    3. Greedily inject exactly the capital needed (`shortfall = (theta * Assets) - CurrentCapital + tiny_epsilon`) moving down the list until `budget` is exhausted.
    4. Run `simulate()` AGAIN with the augmented capital to track saved nodes and new global losses.
* Return a dictionary with `'banks_saved'`, `'capital_spent'`, exact savings metrics, and the returned dictionary from the second simulation (`'sim_after'`).

## 5. Geographic Mapping & Visuals (`utils.py`)
* Load initial banks (`network_input.csv`) and map their cities/countries to `lon, lat`.
* If multiple banks share a city, add a deterministic micro-jitter (`random.uniform(-0.15, 0.15)` seeded uniquely) to their coordinates to prevent overlapping on the map.
* **Plotly Map (`build_plotly_map`):**
    * Use `projection_type = 'orthographic'` (Lon=15, Lat=54) to create a 3D Earth globe.
    * Base Style: No countries/borders visible. Land = `#161622`, Ocean = `#05050A`. 
    * The map background and paper must be completely transparent (`rgba(0,0,0,0)`).
    * Ensure `margin=dict(l=0, r=0, t=0, b=0)` so the map bleeds perfectly.
* **Notion-Style HTML Tooltips:** Use Plotly's `hovertemplate` to display premium HTML overlays. Badges for status: "Healthy" (`#8B5CF6`), "Rescued" (`#06B6D4`), "Defaulted" (`#F43F5E`). Show the bank name, assets, and capital ratio dynamically.

## 6. NEW Premium SaaS UI Architecture (`app.py`)
We are entirely abandoning Streamlit's default container layout through heavy CSS injection.
* **Map = Absolute Background:** Render `st.plotly_chart` using `config={'displayModeBar': False}`. Inject CSS to target `[data-testid="stPlotlyChart"]` assigning `position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 0;`.
* **Dark Gradient Aesthetic:** Assign `html, body` a radial background gradient: `radial-gradient(circle at 15% 50%, rgba(32, 22, 54, 0.6), #05050A 70%)`. Prevent scrolling on `.block-container` by setting padding and margin to 0 and `max-width: 100%`.
* **3-Column Overlay Layout:** Implement `st.columns([1.7, 5, 1.7])`. We target these array elements with CSS `nth-of-type` to create floating panels:
    * **Left Panel (col 1):** `position: relative; z-index: 10;`. Frosted background (`rgba(5, 5, 10, 0.55)`), blurred (`backdrop-filter: blur(20px)`), styling standard Streamlit sliders for parameters (`alpha`, `theta`, `lambda_`, `budget`), and sleek `st.metric` cards to show Live Damages.
    * **Center Panel (col 2):** Set CSS `pointer-events: none` on this column so mouse clicks pass through the vacuum to rotate the 3D globe underneath.
    * **Right Panel (col 3):** `position: relative; z-index: 10;`. A transparent feed containing a generated list of `st.button` inputs for every bank in the network.
* **Loading "Booting" Screen:** While `networkx` initializes, block the UI with an `st.empty()` container showing a full-screen "Antigravity | Beta Booting" placeholder. Remove the container when graph instantiation completes in `st.session_state`.
* **Focus Mode (Bi-directional Selection):** If the user selects a `st.button` in the right-hand panel, isolate the selected bank on the map (fading out all unrelated nodes to opacity `0.08` while keeping the selected node and its first-order neighbors glowing at `0.9`). If they use Plotly's `on_select` to click a node on the globe directly, synchronize the Streamlit selection dynamically.

## 7. Execution Context
Remember to build functions with rigorous input checking. `simulate()` uses strictly typed keyword arguments. Handle dictionary returns carefully (e.g., extracting the final state from the returned `sim_after` dict of the intervention: `df_state = sim_after['final_network_state']` and calculating metrics off the boolean `'defaulted'` column). Rebuild the system elegantly from the ground up prioritizing this aesthetic scale and removing any past technical debt.
