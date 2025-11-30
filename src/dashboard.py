import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import math
import random
import numpy as np

# Adres API
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="WSP Solver - Thesis Visualization", layout="wide")

st.title("🎓 WSP Solver: System Optymalizacji")
st.markdown("Tryb dwuetapowy: **GCP** (Stoły) + **TSP** (Kolejność).")

# --- 1. FUNKCJE GENERUJĄCE ---
def generate_extreme_data(scenario_type, n_guests=50):
    guests = []
    for i in range(n_guests):
        size = random.choice([1, 2, 2, 2, 3, 4]) 
        guests.append({"id": i, "size": size, "name": f"Grupa_{i} ({size}os.)"})
        
    relationships = []
    
    if "Konflikt" in scenario_type:
        for i in range(n_guests):
            for j in range(i + 1, n_guests):
                if random.random() < 0.25: 
                    relationships.append({"source": i, "target": j, "is_conflict": True, "weight": 1000})
    elif "Preferencja" in scenario_type:
        for i in range(n_guests):
            for j in range(i + 1, n_guests):
                if random.random() < 0.35:
                    relationships.append({"source": i, "target": j, "is_conflict": False, "weight": -10})
    else: 
        for _ in range(n_guests * 3):
            u, v = random.sample(range(n_guests), 2)
            w = 1000 if random.random() < 0.2 else -5
            relationships.append({"source": u, "target": v, "is_conflict": (w > 0), "weight": w})
    
    return guests, relationships

# --- 2. WIZUALIZACJA SIECI ---
def draw_global_network(guests, relationships, assignment):
    G = nx.Graph()
    for g in guests: 
        label = f"{g['name']}\n(x{g['size']})"
        G.add_node(g['id'], label=label, size=g['size'])
    
    edge_x_p, edge_y_p = [], [] 
    edge_x_c, edge_y_c = [], [] 

    for r in relationships: 
        G.add_edge(r['source'], r['target'])
        
    pos = nx.spring_layout(G, seed=42, k=0.3)

    for r in relationships:
        if r['source'] not in pos or r['target'] not in pos: continue
        x0, y0 = pos[r['source']]
        x1, y1 = pos[r['target']]
        if r['is_conflict']:
            edge_x_c.extend([x0, x1, None])
            edge_y_c.extend([y0, y1, None])
        else:
            edge_x_p.extend([x0, x1, None])
            edge_y_p.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x_p, y=edge_y_p, line=dict(width=1, color='green'), mode='lines', name='Preferencje'))
    fig.add_trace(go.Scatter(x=edge_x_c, y=edge_y_c, line=dict(width=1.5, color='red'), mode='lines', name='Konflikty'))

    node_x, node_y, node_color, node_text, node_size_viz = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        tid = assignment.get(str(node), 0) if assignment else 0
        node_color.append(tid)
        size = G.nodes[node]['size']
        label = G.nodes[node]['label']
        node_text.append(f"{label}<br>Stół: {tid}")
        node_size_viz.append(15 + (size * 5))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(showscale=True, colorscale='Turbo', size=node_size_viz, color=node_color, line_width=2),
        text=[str(n) for n in G.nodes()], textposition="top center", hovertext=node_text, name="Grupy"
    ))
    
    fig.update_layout(showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,t=30,b=0), height=600)
    return fig

# --- 3. WIZUALIZACJA STOŁÓW (TSP) ---
def draw_round_tables_detail(arrangements, guests_data):
    fig = go.Figure()
    cols = 3
    spacing = 25
    r_table = 6
    colors = px.colors.qualitative.Pastel
    guest_map = {g['id']: g for g in guests_data}

    for idx, (table_id_str, guest_list) in enumerate(arrangements.items()):
        table_id = int(table_id_str)
        if not guest_list: continue
        
        row, col = divmod(idx, cols)
        cx, cy = col * spacing, -row * spacing
        color = colors[idx % len(colors)]
        current_load = sum(guest_map[gid]['size'] for gid in guest_list)
        
        fig.add_shape(type="circle", x0=cx-r_table, y0=cy-r_table, x1=cx+r_table, y1=cy+r_table, fillcolor=color, opacity=0.3, line_color="gray")
        fig.add_annotation(x=cx, y=cy, text=f"Stół {table_id}<br>({current_load} os.)", showarrow=False, font=dict(size=14))

        n = len(guest_list)
        coords = []
        for i, gid in enumerate(guest_list):
            angle = 2 * math.pi * i / n
            x = cx + r_table * 0.8 * math.cos(angle)
            y = cy + r_table * 0.8 * math.sin(angle)
            coords.append((x, y, gid))
        
        if n > 1:
            px_ = [c[0] for c in coords] + [coords[0][0]]
            py_ = [c[1] for c in coords] + [coords[0][1]]
            fig.add_trace(go.Scatter(x=px_, y=py_, mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False))

        for x, y, gid in coords:
            g_size = guest_map[gid]['size']
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(size=20 + (g_size*4), color='white', line=dict(width=2, color='black')),
                text=str(gid), textposition="middle center",
                hovertext=f"{guest_map[gid]['name']} (x{g_size})", showlegend=False
            ))

    fig.update_layout(height=800, plot_bgcolor="white", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# --- SIDEBAR & DATA LOGIC ---
with st.sidebar:
    st.header("🎛️ Panel Sterowania")
    mode = st.radio("Tryb Pracy:", ["Generator Losowy", "Wprowadzanie Ręczne (Matrix)"])
    
    if mode == "Generator Losowy":
        scenario = st.selectbox("Scenariusz", ["Zbalansowany", "Maksymalny Konflikt", "Same Preferencje"])
        n_guests = st.slider("Liczba grup", 10, 200, 30)
        n_tables = st.slider("Liczba stołów", 2, 20, 5)
        cap = st.number_input("Pojemność stołu (osoby)", 2, 30, 10)
        
        if st.button("🎲 Generuj Dane"):
            g, r = generate_extreme_data(scenario, n_guests)
            st.session_state['data'] = {"guests": g, "relationships": r, "num_tables": n_tables, "table_capacity": cap, "source": "random"}
            st.success("Wygenerowano!")

    else:
        st.info("Zdefiniuj grupy i relacje.")
        
        if 'manual_guests' not in st.session_state:
            st.session_state['manual_guests'] = pd.DataFrame([
                {"ID": 0, "Nazwa Grupy": "A", "Rozmiar": 2}, 
                {"ID": 1, "Nazwa Grupy": "B", "Rozmiar": 2}, 
                {"ID": 2, "Nazwa Grupy": "C", "Rozmiar": 2}
            ])
            
        edited_guests = st.data_editor(st.session_state['manual_guests'], num_rows="dynamic", hide_index=True)
        
        current_ids = [str(int(row["ID"])) for _, row in edited_guests.iterrows() if pd.notna(row["ID"])]
        
        if 'matrix_df' not in st.session_state:
             st.session_state['matrix_df'] = pd.DataFrame(0, index=current_ids, columns=current_ids)
        else:
            st.session_state['matrix_df'] = st.session_state['matrix_df'].reindex(index=current_ids, columns=current_ids, fill_value=0)

        st.write("**Macierz Relacji** (1000=Konflikt, -5=Preferencja)")
        edited_matrix = st.data_editor(st.session_state['matrix_df'], key="matrix_editor")
        
        man_tables = st.number_input("Liczba stołów", 1, 10, 2)
        man_cap = st.number_input("Pojemność stołu (osoby)", 1, 50, 10)
        
        if st.button("💾 Zatwierdź Dane"):
            rels_out = []
            guests_out = []
            valid_ids = []
            for _, row in edited_guests.iterrows():
                try:
                    gid = int(row["ID"])
                    valid_ids.append(gid)
                    guests_out.append({"id": gid, "size": int(row["Rozmiar"]), "name": str(row["Nazwa Grupy"])})
                except: pass

            for i, src_id in enumerate(valid_ids):
                for j, tgt_id in enumerate(valid_ids):
                    if i >= j: continue 
                    try:
                        val = edited_matrix.loc[str(src_id), str(tgt_id)]
                        val = int(val) 
                        if val != 0:
                            is_conf = bool(val > 0)
                            rels_out.append({"source": int(src_id), "target": int(tgt_id), "weight": int(val), "is_conflict": is_conf})
                    except: pass
            
            st.session_state['data'] = {
                "guests": guests_out, "relationships": rels_out, "num_tables": man_tables, "table_capacity": man_cap, "source": "manual"
            }
            st.success(f"Zapisano!")

    st.divider()
    alg = st.selectbox("Algorytm (Single Run)", ["tabu_search", "dsatur", "greedy"])
    params = {}
    if alg == "tabu_search":
        params["max_iterations"] = st.slider("Iteracje", 100, 2000, 500)

# --- WIDOK GŁÓWNY ---
if 'data' in st.session_state:
    d = st.session_state['data']
    
    # ZAKŁADKI: Single vs Benchmark
    tab_single, tab_benchmark = st.tabs(["🚀 Pojedyncze Uruchomienie", "📊 Benchmark (Porównanie)"])
    
    with tab_single:
        st.write(f"### Zestaw danych: {len(d['guests'])} grup")
        if st.button("Start Solver"):
            with st.spinner("Liczenie..."):
                payload = {
                    "algorithm": alg, "num_tables": d['num_tables'], 
                    "table_capacity": d['table_capacity'], "guests": d['guests'], 
                    "relationships": d['relationships'], "params": params
                }
                try:
                    res = requests.post(f"{API_URL}/solve", json=payload).json()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Wynik (Score)", f"{res['score']:.1f}")
                    c2.metric("Konflikty", res['conflicts'])
                    c3.metric("Czas", f"{res['time_seconds']:.3f}s")
                    
                    t1, t2 = st.tabs(["Graf Relacji", "Plan Stołów (TSP)"])
                    with t1: st.plotly_chart(draw_global_network(d['guests'], d['relationships'], res['assignment']), use_container_width=True)
                    with t2: st.plotly_chart(draw_round_tables_detail(res['arrangements'], d['guests']), use_container_width=True)
                    
                    if res['history'] and len(res['history']) > 1:
                        df = pd.DataFrame(res['history'])
                        fig = px.line(df, x='iteration', y='score', title="Konwergencja")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(str(e))

    with tab_benchmark:
        st.write("Porównanie wydajności na aktualnych danych.")
        bench_iter = st.slider("Iteracje TS (Benchmark)", 100, 2000, 500, key="bench_iter")
        
        if st.button("Uruchom Benchmark"):
            results = {}
            algs_to_check = ["greedy", "dsatur", "tabu_search"]
            progress_bar = st.progress(0)
            
            for i, a in enumerate(algs_to_check):
                p = {**d, "algorithm": a, "guests": d['guests'], "relationships": d['relationships'], "params": {"max_iterations": bench_iter}}
                try:
                    results[a] = requests.post(f"{API_URL}/solve", json=p).json()
                except: pass
                progress_bar.progress((i + 1) / len(algs_to_check))
            
            # WYKRES LINIOWY PORÓWNAWCZY
            st.subheader("Wyścig Algorytmów: Score vs Iteracje")
            fig_bench = go.Figure()
            
            # TS (Linia zmienna)
            ts_data = results.get("tabu_search", {}).get("history", [])
            if ts_data:
                df_ts = pd.DataFrame(ts_data)
                fig_bench.add_trace(go.Scatter(x=df_ts['iteration'], y=df_ts['score'], mode='lines', name='Tabu Search', line=dict(color='firebrick', width=3)))
            
            # Greedy i DSatur (Linie poziome)
            x_axis = [0, bench_iter]
            if "greedy" in results:
                s = results["greedy"]["score"]
                fig_bench.add_trace(go.Scatter(x=x_axis, y=[s, s], mode='lines', name=f'Greedy ({s:.1f})', line=dict(color='royalblue', width=2)))
            if "dsatur" in results:
                s = results["dsatur"]["score"]
                fig_bench.add_trace(go.Scatter(x=x_axis, y=[s, s], mode='lines', name=f'DSatur ({s:.1f})', line=dict(color='forestgreen', width=2)))

            fig_bench.update_layout(xaxis_title="Iteracje", yaxis_title="Score (Mniej=Lepiej)", height=500)
            st.plotly_chart(fig_bench, use_container_width=True)
            
            summary_data = []
            for k, v in results.items():
                summary_data.append({"Algorytm": k.upper(), "Score": v['score'], "Czas (s)": v['time_seconds'], "Konflikty": v['conflicts']})
            st.dataframe(pd.DataFrame(summary_data))

else:
    st.info("👈 Skonfiguruj dane w menu po lewej.")
