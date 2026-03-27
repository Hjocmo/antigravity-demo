"""
utils.py — Data Helpers and Graph Drawing Utilities
====================================================
Provides data loading, mock-data fallback, and Plotly/NetworkX
visualisation helpers for the Streamlit UI.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_nodes(csv_path: str = "network_input_geocoded.csv") -> pd.DataFrame:
    """
    Load the bank-node DataFrame from network_input.csv.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with columns:
        LEI_Code, BankName, Country, TotalAssets, TotalAssets_norm,
        CR, M_i_final, FI.
    """
    df = pd.read_csv(csv_path, on_bad_lines='warn')

    # Ensure required columns exist
    required = ['BankName', 'Country', 'TotalAssets', 'CR', 'M_i_final']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"network_input.csv is missing columns: {missing}")

    # Fill optional columns with sensible defaults if absent
    if 'TotalAssets_norm' not in df.columns:
        max_a = df['TotalAssets'].max()
        df['TotalAssets_norm'] = df['TotalAssets'] / max_a if max_a > 0 else 1.0

    if 'FI' not in df.columns:
        df['FI'] = 0.0

    if 'LEI_Code' not in df.columns:
        df['LEI_Code'] = [f"LEI_{i:04d}" for i in range(len(df))]

    # Add a deterministic micro-jitter (+/- degrees) to lat and lon
    # to prevent perfectly overlapping banks in the same city.
    if 'lat' in df.columns and 'lon' in df.columns:
        np.random.seed(42)
        df['lat'] = df['lat'] + np.random.uniform(-0.5, 0.5, size=len(df))
        df['lon'] = df['lon'] + np.random.uniform(-0.5, 0.5, size=len(df))

    return df.reset_index(drop=True)


def get_unique_countries(df_nodes: pd.DataFrame) -> list[str]:
    """
    Return a sorted list of unique country codes present in the dataset.

    Parameters
    ----------
    df_nodes : pd.DataFrame

    Returns
    -------
    list[str]
    """
    return sorted(df_nodes['Country'].dropna().unique().tolist())





# ---------------------------------------------------------------------------
# Graph Drawing
# ---------------------------------------------------------------------------

def build_plotly_network(
    G: nx.DiGraph,
    df_state: pd.DataFrame,
    title: str = "Interbank Contagion Network",
) -> go.Figure:
    """
    Build an interactive Plotly figure of the interbank network.

    Node colour encoding
    --------------------
    - Red   : defaulted bank (not rescued).
    - Green : bank rescued by intervention (rescued == True and not defaulted).
    - Blue  : healthy / surviving bank.

    Node size is proportional to log(TotalAssets).

    Parameters
    ----------
    G : nx.DiGraph
        The interbank network graph (from engine.build_network).
    df_state : pd.DataFrame
        DataFrame with at minimum: BankName, TotalAssets, defaulted (bool),
        and optionally `rescued` (bool) for the post-intervention graph.
    title : str
        Figure title displayed in the Plotly layout.

    Returns
    -------
    go.Figure
        A Plotly Figure ready to be rendered with `st.plotly_chart()`.
    """
    # ------------------------------------------------------------------
    # Layout — spring layout gives a natural force-directed look
    # ------------------------------------------------------------------
    pos = nx.spring_layout(G, seed=42, k=0.6)

    # ------------------------------------------------------------------
    # Edge traces — thin grey arrows
    # ------------------------------------------------------------------
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        if src in pos and dst in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.4, color='rgba(180,180,200,0.4)'),
        hoverinfo='none',
        showlegend=False,
    )

    # ------------------------------------------------------------------
    # Node traces — one per status so the legend is meaningful
    # ------------------------------------------------------------------
    df_indexed = df_state.reset_index(drop=True)

    has_rescued = 'rescued' in df_indexed.columns

    # Categorise each node
    node_groups = {
        'Defaulted':  {'color': '#EF4444', 'symbol': 'circle', 'nodes': []},
        'Rescued':    {'color': '#22C55E', 'symbol': 'star',   'nodes': []},
        'Healthy':    {'color': '#3B82F6', 'symbol': 'circle', 'nodes': []},
    }

    for node_id in G.nodes():
        if node_id >= len(df_indexed):
            continue
        row = df_indexed.iloc[node_id]
        defaulted = bool(row.get('defaulted', False))
        rescued   = has_rescued and bool(row.get('rescued', False))

        if rescued and not defaulted:
            node_groups['Rescued']['nodes'].append(node_id)
        elif defaulted:
            node_groups['Defaulted']['nodes'].append(node_id)
        else:
            node_groups['Healthy']['nodes'].append(node_id)

    node_traces = []
    for label, cfg in node_groups.items():
        if not cfg['nodes']:
            continue

        nx_arr, ny_arr, sizes, texts = [], [], [], []
        for nid in cfg['nodes']:
            if nid not in pos:
                continue
            row = df_indexed.iloc[nid]
            x, y = pos[nid]
            nx_arr.append(x)
            ny_arr.append(y)

            assets = float(row.get('TotalAssets', 1))
            size = max(8, min(30, 6 + 3.5 * np.log10(max(assets, 1))))
            sizes.append(size)

            bank  = row.get('BankName', f'Bank {nid}')
            cr    = row.get('CR', float('nan'))
            c_fin = row.get('C_final', float('nan'))
            hover = (
                f"<b>{bank}</b><br>"
                f"Country: {row.get('Country','?')}<br>"
                f"Total Assets: €{assets/1e3:.1f}B<br>"
                f"Cap. Ratio: {cr:.3f}<br>"
                f"Final Capital: €{c_fin/1e3:.1f}B<br>"
                f"Status: <b>{label}</b>"
            )
            texts.append(hover)

        node_traces.append(go.Scatter(
            x=nx_arr, y=ny_arr,
            mode='markers',
            marker=dict(
                size=sizes,
                color=cfg['color'],
                symbol=cfg['symbol'],
                line=dict(width=1.2, color='rgba(255,255,255,0.7)'),
                opacity=0.92,
            ),
            text=texts,
            hoverinfo='text',
            name=label,
        ))

    # ------------------------------------------------------------------
    # Compose figure
    # ------------------------------------------------------------------
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=15, color='#E2E8F0'), x=0.5, xanchor='center'),
            paper_bgcolor='#0F172A',
            plot_bgcolor='#0F172A',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(30,41,59,0.85)',
                bordercolor='rgba(148,163,184,0.3)',
                borderwidth=1,
                font=dict(color='#CBD5E1', size=11),
            ),
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=45),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )
    return fig


def build_plotly_map(
    G: nx.DiGraph,
    df_state: pd.DataFrame,
    title: str = "Interbank Geographic Map",
    selected_bank_name: str | None = None,
) -> go.Figure:
    """
    Build a Geographic map of Europe with banks plotted at their approximate
    country coordinates using go.Scattergeo, including interbank edges.
    Supports a Focus Mode to highlight a specific bank and its neighbors.
    """
    df_indexed = df_state.reset_index(drop=True)
    has_rescued = 'rescued' in df_indexed.columns

    target_idx = None
    neighbor_indices = set()
    if selected_bank_name:
        matches = df_indexed.index[df_indexed['BankName'] == selected_bank_name].tolist()
        if matches:
            target_idx = matches[0]
            # Get undirected neighbors (both inbound and outbound exposures)
            neighbor_indices = set(G.predecessors(target_idx)).union(set(G.successors(target_idx)))
            title += f"<br><span style='font-size:12px;color:#FACC15;'>Highlighting {selected_bank_name} and its {len(neighbor_indices)} interbank connections</span>"
        else:
            title += f"<br><span style='font-size:12px;color:#EF4444;'>Error: '{selected_bank_name}' not found in network nodes!</span>"

    node_groups = {
        'Defaulted':  {'color': '#F43F5E', 'symbol': 'circle', 'data': []},
        'Rescued':    {'color': '#06B6D4', 'symbol': 'circle',   'data': []},
        'Healthy':    {'color': '#8B5CF6', 'symbol': 'circle', 'data': []},
    }

    for idx, row in df_indexed.iterrows():
        defaulted = bool(row.get('defaulted', False))
        rescued   = has_rescued and bool(row.get('rescued', False))
        
        status = ""
        if rescued and not defaulted:
            status = 'Rescued'
        elif defaulted:
            status = 'Defaulted'
        else:
            status = 'Healthy'

        assets = float(row.get('TotalAssets', 1))
        size = max(5, min(24, 4 + 2.5 * np.log10(max(assets, 1))))
        
        bank  = row.get('BankName', 'Unknown Bank')
        cr    = row.get('CR', float('nan'))
        c_fin = row.get('C_final', float('nan'))

        # Evaluate notion-style Hover Tooltip
        badge_bg = "#8B5CF6" if status == "Healthy" else ("#06B6D4" if status == "Rescued" else "#F43F5E")
        hover_html = f"""<div style='font-family:"Inter",sans-serif;line-height:1.4;padding:4px;'>
            <span style='background-color:{badge_bg};color:#fff;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;text-transform:uppercase;'>{status}</span><br><br>
            <span style='font-size:15px;font-weight:600;color:#fff;'>🏦 {bank}</span><br>
            <span style='color:#94a3b8;font-size:12px;'>{row.get('Country','?')} — Assets: €{assets/1e3:.1f}B</span><br>
            <span style='color:#94a3b8;font-size:12px;'>CR: <b style='color:#fff;'>{cr:.1%}</b></span>
        </div>"""
        
        item = {
            'lon': row.get('lon'), 
            'lat': row.get('lat'), 
            'size': size, 
            'bank': bank, 
            'status': status,
            'hovertext': hover_html,
            'original_idx': idx # Store original index for focus mode
        }
        
        node_groups[status]['data'].append(item)

    traces = []
    for group_name, group_info in node_groups.items():
        if not group_info['data']:
            continue
            
        hl_items = []
        faded_items = []
        
        for item in group_info['data']:
            nid = item['original_idx']
            # Apply focus separation
            if target_idx is not None:
                if nid == target_idx or nid in neighbor_indices:
                    hl_items.append(item)
                else:
                    faded_items.append(item)
            else:
                hl_items.append(item)

        # FADED Nodes Trace
        if faded_items:
            traces.append(go.Scattergeo(
                lon=[x['lon'] for x in faded_items],
                lat=[x['lat'] for x in faded_items],
                mode='markers',
                marker=dict(size=10, symbol=group_info['symbol'], color=group_info['color'], line=dict(width=0.5, color='rgba(255, 255, 255, 0.2)')), # Less prominent borders
                text=[x['hovertext'] for x in faded_items],
                hovertemplate='%{text}<extra></extra>',
                hoverlabel=dict(bgcolor='rgba(15, 15, 20, 0.95)', bordercolor='rgba(255, 255, 255, 0.1)', font=dict(family='Inter')),
                name=group_name + " (faded)",
                customdata=[x['bank'] for x in faded_items],
                opacity=0.08,  # Heavily faded
                showlegend=False
            ))
            
        # HIGHLIGHTED/NORMAL Nodes Trace
        if hl_items:
            traces.append(go.Scattergeo(
                lon=[x['lon'] for x in hl_items],
                lat=[x['lat'] for x in hl_items],
                mode='markers',
                marker=dict(size=14, symbol=group_info['symbol'], color=group_info['color'], line=dict(width=1, color='rgba(255, 255, 255, 0.8)')), # Increased size and border
                text=[x['hovertext'] for x in hl_items],
                hovertemplate='%{text}<extra></extra>',
                hoverlabel=dict(bgcolor='rgba(15, 15, 20, 0.95)', bordercolor='rgba(255, 255, 255, 0.1)', font=dict(family='Inter')),
                name=group_name,
                customdata=[x['bank'] for x in hl_items],
                opacity=0.9,
                showlegend=True,
            ))

    # Add edges
    normal_edge_lons, normal_edge_lats = [], []
    highlight_edge_lons, highlight_edge_lats = [], []
    
    for src, dst in G.edges():
        if src < len(df_indexed) and dst < len(df_indexed):
            src_row = df_indexed.iloc[src]
            dst_row = df_indexed.iloc[dst]
            
            is_highlighted = False
            if target_idx is not None:
                if (src == target_idx and dst in neighbor_indices) or (dst == target_idx and src in neighbor_indices):
                    is_highlighted = True
                    
            if is_highlighted:
                highlight_edge_lons.extend([src_row.get('lon'), dst_row.get('lon'), None])
                highlight_edge_lats.extend([src_row.get('lat'), dst_row.get('lat'), None])
            else:
                normal_edge_lons.extend([src_row.get('lon'), dst_row.get('lon'), None])
                normal_edge_lats.extend([src_row.get('lat'), dst_row.get('lat'), None])

    # Add highlighted edges on top
    if highlight_edge_lons:
        traces.insert(0, go.Scattergeo(
            lon=highlight_edge_lons,
            lat=highlight_edge_lats,
            mode='lines',
            line=dict(width=1.5, color='rgba(139, 92, 246, 0.8)'), # Neon Violet
            hoverinfo='none',
            showlegend=False,
        ))

    # Add normal edges beneath (faded heavily if focus mode is active)
    if normal_edge_lons:
        normal_opacity = 0.02 if target_idx is not None else 0.15
        traces.insert(0, go.Scattergeo(
            lon=normal_edge_lons,
            lat=normal_edge_lats,
            mode='lines',
            line=dict(width=0.2, color=f'rgba(255, 255, 255, {normal_opacity})'),
            hoverinfo='none',
            showlegend=False,
        ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=14, color='#F8FAFC', family='Inter'), x=0.5, xanchor='center'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(5, 5, 10, 0.85)',
                bordercolor='rgba(255, 255, 255, 0.1)',
                borderwidth=1,
                font=dict(color='#8B95A5', size=11, family='Inter'),
            ),
            geo=dict(
                scope='europe',
                showland=True,
                landcolor='#111827',
                showocean=True,
                oceancolor='#010119',
                bgcolor='rgba(0,0,0,0)',
                showcountries=True,
                countrycolor='rgba(255,255,255,0.05)',
                coastlinecolor='rgba(255,255,255,0.1)',
                showlakes=False,
                resolution=50
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=550
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_losses(value: float) -> str:
    """
    Format a monetary loss value (EUR millions) for human-readable display.

    Examples
    --------
    >>> format_losses(1_250_000)
    '€1.25T'
    >>> format_losses(45_000)
    '€45.0B'

    Parameters
    ----------
    value : float
        Loss in EUR millions.

    Returns
    -------
    str
    """
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.2f}T"
    if value >= 1_000:
        return f"€{value / 1_000:.1f}B"
    return f"€{value:.0f}M"
