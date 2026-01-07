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

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="WSP Solver - Thesis Visualization", layout="wide")

st.title("WSP Solver: System Optymalizacji")
st.markdown("Tryb dwuetapowy: **GCP** (Stoły) + **TSP** (Kolejność).")


def generate_extreme_data(scenario_type, total_people=100, num_groups=None):
    # Auto-obliczenie liczby grup
    if num_groups is None:
        num_groups = max(10, total_people // 3)
    
    # Walidacja
    if total_people < num_groups:
        st.error(f"Błąd: {total_people} osób na {num_groups} grup - niemożliwe!")
        return [], []
    
    group_sizes = [1] * num_groups  # Start: każda grupa ma 1 osobę
    remaining = total_people - num_groups
    max_group_size = 6
    
    while remaining > 0:
        idx = random.randint(0, num_groups - 1)
        if group_sizes[idx] < max_group_size:
            group_sizes[idx] += 1
            remaining -= 1
        elif all(s >= max_group_size for s in group_sizes):
            group_sizes[idx] += 1
            remaining -= 1
    
    random.shuffle(group_sizes)
    
    # Tworzenie obiektów gości
    guests = []
    for i, size in enumerate(group_sizes):
        guests.append({
            "id": i,
            "size": size,
            "name": f"Grupa_{i} ({size}os.)"
        })
    
    relationships = []
    
    # Dostosuj prawdopodobieństwa do scenariusza
    if "Konflikt" in scenario_type:
        conflict_prob = 0.25
        preference_prob = 0.10
    elif "Preferencja" in scenario_type:
        conflict_prob = 0.05
        preference_prob = 0.35
    else:  # Zbalansowany
        conflict_prob = 0.15
        preference_prob = 0.20
    
    # Generuj relacje dla każdej pary grup
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            rand = random.random()
            
            if rand < conflict_prob:
                relationships.append({
                    "source": i,
                    "target": j,
                    "is_conflict": True,
                    "weight": 1000
                })
            elif rand < conflict_prob + preference_prob:
                relationships.append({
                    "source": i,
                    "target": j,
                    "is_conflict": False,
                    "weight": -5
                })
    
    return guests, relationships


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
        if r['source'] not in pos or r['target'] not in pos: 
            continue
        x0, y0 = pos[r['source']]
        x1, y1 = pos[r['target']]
        if r['is_conflict']:
            edge_x_c.extend([x0, x1, None])
            edge_y_c.extend([y0, y1, None])
        else:
            edge_x_p.extend([x0, x1, None])
            edge_y_p.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x_p, y=edge_y_p, 
        line=dict(width=1, color='green'), 
        mode='lines', name='Preferencje'
    ))
    fig.add_trace(go.Scatter(
        x=edge_x_c, y=edge_y_c, 
        line=dict(width=1.5, color='red'), 
        mode='lines', name='Konflikty'
    ))

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
        marker=dict(
            showscale=True, 
            colorscale='Turbo', 
            size=node_size_viz, 
            color=node_color, 
            line_width=2
        ),
        text=[str(n) for n in G.nodes()], 
        textposition="top center", 
        hovertext=node_text, 
        name="Grupy"
    ))
    
    fig.update_layout(
        showlegend=True, 
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        margin=dict(l=0, r=0, t=30, b=0), 
        height=600
    )
    return fig


def draw_round_tables_detail(arrangements, guests_data):
    fig = go.Figure()
    cols = 3
    spacing = 25
    r_table = 6
    colors = px.colors.qualitative.Pastel
    guest_map = {g['id']: g for g in guests_data}

    for idx, (table_id_str, guest_list) in enumerate(arrangements.items()):
        table_id = int(table_id_str)
        if not guest_list: 
            continue
        
        row, col = divmod(idx, cols)
        cx, cy = col * spacing, -row * spacing
        color = colors[idx % len(colors)]
        current_load = sum(guest_map[gid]['size'] for gid in guest_list)
        
        fig.add_shape(
            type="circle", 
            x0=cx - r_table, y0=cy - r_table, 
            x1=cx + r_table, y1=cy + r_table, 
            fillcolor=color, opacity=0.3, line_color="gray"
        )
        fig.add_annotation(
            x=cx, y=cy, 
            text=f"Stół {table_id}<br>({current_load} os.)", 
            showarrow=False, font=dict(size=14)
        )

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
            fig.add_trace(go.Scatter(
                x=px_, y=py_, mode='lines', 
                line=dict(color='gray', width=1, dash='dot'), 
                showlegend=False
            ))

        for x, y, gid in coords:
            g_size = guest_map[gid]['size']
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(
                    size=20 + (g_size * 4), 
                    color='white', 
                    line=dict(width=2, color='black')
                ),
                text=str(gid), textposition="middle center",
                hovertext=f"{guest_map[gid]['name']} (x{g_size})", 
                showlegend=False
            ))

    fig.update_layout(
        height=800, 
        plot_bgcolor="white", 
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False)
    )
    return fig

with st.sidebar:
    st.header("Panel Sterowania")
    mode = st.radio(
        "Tryb Pracy:", 
        ["Generator Losowy", "Wprowadzanie Ręczne (Matrix)"]
    )
    
    if mode == "Generator Losowy":
        st.subheader("Parametry Instancji")
        
        # KROK 1: GŁÓWNY PARAMETR - Liczba osób (nie grup!)
        total_people = st.slider(
            "Całkowita liczba osób", 
            min_value=20, 
            max_value=500, 
            value=100,
            step=10,
            help="Łączna liczba gości weselnych (główny parametr problemu)"
        )
        
        # KROK 2: Liczba grup (opcjonalnie, z auto-kalkulacją)
        default_groups = max(10, total_people // 3)
        num_groups = st.slider(
            "Liczba grup", 
            min_value=max(5, total_people // 6),  # Min: grupy 6-osobowe
            max_value=total_people,                # Max: grupy 1-osobowe
            value=default_groups,
            help=f"System wygeneruje grupy o rozmiarach 1-6 osób. "
                 f"Domyślnie: {default_groups} grup (~{total_people/default_groups:.1f} os./grupa)"
        )
        
        # KROK 3: Konfiguracja sali (AUTO-OBLICZANA!)
        table_capacity = st.slider("Pojemność stołu", 8, 14, 10)
        auto_tables = math.ceil(total_people / table_capacity)
        
        st.info(
            f"**Rekomendacja**: {auto_tables} stołów "
            f"(dla {total_people} osób i pojemności {table_capacity})"
        )
        
        n_tables = st.slider(
            "Liczba stołów", 
            min_value=auto_tables,        # Minimum: wystarczy dla wszystkich
            max_value=auto_tables + 5,
            value=auto_tables,
            help="Możesz dodać rezerwowe stoły (ale to zwiększy karę za balans)"
        )
        
        # KROK 4: Scenariusz relacji
        scenario = st.selectbox(
            "Scenariusz relacji",
            ["Zbalansowany", "Maksymalny Konflikt", "Same Preferencje"]
        )
        
        st.divider()
        
        # PRZYCISK GENEROWANIA
        if st.button("Generuj Dane Testowe", type="primary"):
            with st.spinner("Generowanie..."):
                g, r = generate_extreme_data(
                    scenario_type=scenario,
                    total_people=total_people,
                    num_groups=num_groups
                )
                
                if g:  # Jeśli generacja się powiodła
                    st.session_state['data'] = {
                        "guests": g,
                        "relationships": r,
                        "num_tables": n_tables,
                        "table_capacity": table_capacity,
                        "source": "random",
                        # Metadane do wyświetlenia
                        "metadata": {
                            "total_people": total_people,
                            "num_groups": len(g),
                            "avg_group_size": total_people / len(g),
                            "num_conflicts": sum(1 for rel in r if rel['is_conflict']),
                            "num_preferences": sum(1 for rel in r if not rel['is_conflict'])
                        }
                    }
                    st.success(f"Wygenerowano {len(g)} grup ({total_people} osób)!")
                    st.balloons()

    else:
        st.info("Zdefiniuj grupy i relacje ręcznie.")
        
        if 'manual_guests' not in st.session_state:
            st.session_state['manual_guests'] = pd.DataFrame([
                {"ID": 0, "Nazwa Grupy": "A", "Rozmiar": 2}, 
                {"ID": 1, "Nazwa Grupy": "B", "Rozmiar": 2}, 
                {"ID": 2, "Nazwa Grupy": "C", "Rozmiar": 3}
            ])
            
        edited_guests = st.data_editor(
            st.session_state['manual_guests'], 
            num_rows="dynamic", 
            hide_index=True
        )
        
        current_ids = [
            str(int(row["ID"])) 
            for _, row in edited_guests.iterrows() 
            if pd.notna(row["ID"])
        ]
        
        if 'matrix_df' not in st.session_state:
            st.session_state['matrix_df'] = pd.DataFrame(
                0, index=current_ids, columns=current_ids
            )
        else:
            st.session_state['matrix_df'] = st.session_state['matrix_df'].reindex(
                index=current_ids, columns=current_ids, fill_value=0
            )

        st.write("**Macierz Relacji** (1000=Konflikt, -5=Preferencja)")
        edited_matrix = st.data_editor(
            st.session_state['matrix_df'], 
            key="matrix_editor"
        )
        
        # Parametry sali (z auto-kalkulacją)
        total_manual_people = edited_guests["Rozmiar"].sum()
        st.metric("Całkowita liczba osób", int(total_manual_people))
        
        man_cap = st.number_input("Pojemność stołu (osoby)", 4, 20, 10)
        auto_man_tables = math.ceil(total_manual_people / man_cap)
        
        man_tables = st.number_input(
            f"Liczba stołów (min: {auto_man_tables})", 
            auto_man_tables, 20, auto_man_tables
        )
        
        if st.button("Zatwierdź Dane Ręczne", type="primary"):
            rels_out = []
            guests_out = []
            valid_ids = []
            
            for idx, row in edited_guests.iterrows():
                try:
                    # Sprawdź, czy pola nie są puste
                    if pd.isna(row["ID"]) or pd.isna(row["Rozmiar"]):
                        st.warning(f"Wiersz {idx}: Pominięto - brak ID lub Rozmiaru")
                        continue
                    
                    # Konwersja z wymuszeniem na natywne typy Python
                    gid = int(row["ID"])
                    size = int(row["Rozmiar"])
                    name = str(row["Nazwa Grupy"]) if pd.notna(row["Nazwa Grupy"]) else f"Grupa {gid}"
                    
                    # Walidacja zakresu
                    if size < 1:
                        st.warning(f"Grupa {gid}: Rozmiar musi być >= 1 (podano {size})")
                        continue
                    
                    valid_ids.append(gid)
                    guests_out.append({
                        "id": gid,
                        "size": size,
                        "name": name
                    })
                    
                except (ValueError, TypeError) as e:
                    st.warning(f"Wiersz {idx}: Błąd konwersji - {str(e)}")
                    continue

            # Sprawdź, czy są jakieś goście
            if not guests_out:
                st.error("Brak poprawnych danych gości! Wypełnij tabelę powyżej.")
                st.stop()

            # ===== KROK 2: WALIDACJA I KONWERSJA RELACJI =====
            for i, src_id in enumerate(valid_ids):
                for j, tgt_id in enumerate(valid_ids):
                    if i >= j:  # Tylko górny trójkąt macierzy
                        continue
                    
                    try:
                        #wymuś konwersję wartości z macierzy
                        val_raw = edited_matrix.loc[str(src_id), str(tgt_id)]
                        
                        # Obsługa różnych typów zwracanych przez Pandas
                        if pd.isna(val_raw):
                            val = 0
                        elif isinstance(val_raw, (pd.Series, np.ndarray)):
                            val = int(val_raw.iloc[0])  # Weź pierwszy element
                        else:
                            val = int(val_raw)
                        
                        # Dodaj tylko niezerowe relacje
                        if val != 0:
                            rels_out.append({
                                "source": src_id,
                                "target": tgt_id,
                                "weight": val,
                                "is_conflict": bool(val > 0)  # Wymuś bool
                            })
                    
                    except (ValueError, TypeError, KeyError) as e:
                        # Ignoruj błędy w macierzy (np. brakujące indeksy)
                        continue

            total_manual_people = sum(g['size'] for g in guests_out)
            
            st.session_state['data'] = {
                "guests": guests_out,
                "relationships": rels_out,
                "num_tables": man_tables,
                "table_capacity": man_cap,
                "source": "manual",
                "metadata": {
                    "total_people": total_manual_people,
                    "num_groups": len(guests_out),
                    "avg_group_size": total_manual_people / len(guests_out) if guests_out else 0,
                    "num_conflicts": sum(1 for r in rels_out if r['is_conflict']),
                    "num_preferences": sum(1 for r in rels_out if not r['is_conflict'])
                }
            }
            
            st.success(f"Zapisano {len(guests_out)} grup i {len(rels_out)} relacji!")
            
            # Debug info (możesz wyłączyć przed obroną)
            with st.expander("Podgląd danych (debug)"):
                st.write(f"**Przykładowy gość:** {guests_out[0] if guests_out else 'brak'}")
                st.write(f"**Przykładowa relacja:** {rels_out[0] if rels_out else 'brak'}")

    st.divider()
    st.subheader("Parametry Algorytmu")
    
    alg = st.selectbox(
        "Algorytm (Single Run)", 
        ["tabu_search", "dsatur", "greedy"],
        format_func=lambda x: {
            "tabu_search": "Tabu Search",
            "dsatur": "DSatur",
            "greedy": "Greedy"
        }[x]
    )
    
    params = {}
    if alg == "tabu_search":
        params["max_iterations"] = st.slider(
            "Max iteracji", 100, 2000, 500,
            help="Limit iteracji (500-1000 dla małych instancji)"
        )
        params["tabu_tenure"] = st.slider(
            "Tabu tenure", 5, 50, 10,
            help="Rozmiar listy Tabu (pamięć zakazanych ruchów)"
        )

if 'data' in st.session_state:
    d = st.session_state['data']
    meta = d.get('metadata', {})
    
    st.subheader("Statystyki Problemu")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Liczba Osób", 
            meta.get('total_people', sum(g['size'] for g in d['guests']))
        )
    with col2:
        st.metric("Liczba Grup", meta.get('num_groups', len(d['guests'])))
    with col3:
        avg_size = meta.get('avg_group_size', 0) or (
            sum(g['size'] for g in d['guests']) / len(d['guests'])
        )
        st.metric("Średni rozmiar", f"{avg_size:.1f}")
    with col4:
        st.metric("Konflikty", meta.get('num_conflicts', 0))
    with col5:
        st.metric("Preferencje", meta.get('num_preferences', 0))
    
    total_cap = d['num_tables'] * d['table_capacity']
    total_ppl = sum(g['size'] for g in d['guests'])
    utilization = (total_ppl / total_cap * 100) if total_cap > 0 else 0
    
    if utilization > 100:
        st.error(f"**Za mało miejsc!** Goście: {total_ppl}, Pojemność: {total_cap}")
    else:
        st.info(
            f"Pojemność sali: {total_cap} miejsc "
            f"({d['num_tables']} stołów × {d['table_capacity']} osób) | "
            f"Wykorzystanie: **{utilization:.1f}%**"
        )
    
    st.divider()
    
    tab_single, tab_benchmark = st.tabs([
        "Pojedyncze Uruchomienie", 
        "Benchmark (Porównanie)"
    ])
    
    with tab_single:
        if st.button("Start Solver", type="primary", use_container_width=True):
            with st.spinner(f"Optymalizacja algorytmem {alg.upper()}..."):
                payload = {
                    "algorithm": alg,
                    "num_tables": d['num_tables'],
                    "table_capacity": d['table_capacity'],
                    "guests": d['guests'],
                    "relationships": d['relationships'],
                    "params": params
                }
                
                try:
                    response = requests.post(
                        f"{API_URL}/solve", 
                        json=payload, 
                        timeout=30
                    )
                    
                    if response.status_code == 422:
                        st.error("**Błąd walidacji danych (422)**")
                        
                        error_detail = response.json()
                        
                        # Wyświetl czytelnie błędy FastAPI/Pydantic
                        if "detail" in error_detail:
                            details = error_detail["detail"]
                            
                            if isinstance(details, list):
                                st.write("**Znalezione błędy w danych:**")
                                for err in details:
                                    field = " → ".join(str(x) for x in err.get("loc", []))
                                    msg = err.get("msg", "Nieznany błąd")
                                    st.markdown(f"- `{field}`: {msg}")
                            else:
                                st.write(details)
                        
                        # Pokaż pełny JSON w expanderze
                        with st.expander("Pełny JSON odpowiedzi API"):
                            st.json(error_detail)
                        
                        # Pomoc dla użytkownika
                        st.info(
                            "**Wskazówka**: Sprawdź, czy:\n"
                            "- Wszystkie pola w tabeli gości są wypełnione\n"
                            "- Rozmiary grup są liczbami >= 1\n"
                            "- Macierz relacji zawiera tylko liczby całkowite\n"
                            "- Nie ma pustych komórek w kluczowych polach"
                        )
                        st.stop()
                    
                    elif response.status_code != 200:
                        st.error(f"❌ Błąd API: {response.status_code}")
                        st.code(response.text)
                        st.stop()
                    
                    # KONTYNUACJA (jeśli OK)
                    res = response.json()
                    
                    # Wyniki
                    st.success("Optymalizacja zakończona!")
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        score_color = "inverse" if res['score'] < 0 else "off"
                        st.metric(
                            "Wynik (Score)", 
                            f"{res['score']:.1f}",
                            delta=None,
                            delta_color=score_color
                        )
                    with c2:
                        st.metric(
                            "Konflikty", 
                            res['conflicts'],
                            delta=None,
                            delta_color="inverse" if res['conflicts'] == 0 else "off"
                        )
                    with c3:
                        st.metric(
                            "Czas", 
                            f"{res['time_seconds']*1000:.2f} ms"
                        )
                    
                    # Wizualizacje
                    t1, t2 = st.tabs(["Graf Relacji", "Plan Stołów (TSP)"])
                    
                    with t1:
                        st.plotly_chart(
                            draw_global_network(
                                d['guests'], 
                                d['relationships'], 
                                res['assignment']
                            ),
                            use_container_width=True
                        )
                    
                    with t2:
                        st.plotly_chart(
                            draw_round_tables_detail(
                                res['arrangements'], 
                                d['guests']
                            ),
                            use_container_width=True
                        )
                    
                    # Wykres konwergencji (tylko dla Tabu Search)
                    if res.get('history') and len(res['history']) > 1:
                        st.subheader("Konwergencja algorytmu")
                        df = pd.DataFrame(res['history'])
                        fig = px.line(
                            df, x='iteration', y='score',
                            title="Zmiany Score w czasie optymalizacji",
                            labels={'iteration': 'Iteracja', 'score': 'Score'}
                        )
                        fig.add_hline(
                            y=0, line_dash="dash", 
                            line_color="green",
                            annotation_text="Score = 0 (ideał)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except requests.exceptions.Timeout:
                    st.error("**Timeout** - algorytm przekroczył limit 30s")
                    st.info("Spróbuj zmniejszyć liczbę iteracji lub użyć algorytmu Greedy/DSatur")
                    
                except requests.exceptions.ConnectionError:
                    st.error("**Błąd połączenia** - API nie odpowiada")
                    st.info(f"Sprawdź, czy backend działa: `{API_URL}`")
                    
                except Exception as e:
                    st.error(f"**Nieoczekiwany błąd**: {str(e)}")
                    with st.expander("Stack trace (dla debugowania)"):
                        import traceback
                        st.code(traceback.format_exc())

    with tab_benchmark:
        st.write("Porównanie wydajności trzech algorytmów na aktualnych danych.")
        
        bench_iter = st.slider(
            "Iteracje TS (Benchmark)", 
            100, 2000, 500, 
            key="bench_iter",
            help="Limit iteracji dla Tabu Search w trybie porównawczym"
        )
        
        if st.button(
            "Uruchom Benchmark", 
            type="primary", 
            use_container_width=True
        ):
            results = {}
            algs_to_check = ["greedy", "dsatur", "tabu_search"]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, a in enumerate(algs_to_check):
                status_text.text(f"Testowanie: {a.upper()}...")
                
                p = {
                    "algorithm": a,
                    "num_tables": d['num_tables'],
                    "table_capacity": d['table_capacity'],
                    "guests": d['guests'],
                    "relationships": d['relationships'],
                    "params": {"max_iterations": bench_iter}
                }
                
                try:
                    results[a] = requests.post(
                        f"{API_URL}/solve", 
                        json=p, 
                        timeout=60
                    ).json()
                except Exception as e:
                    st.warning(f"{a.upper()} - błąd: {str(e)}")
                
                progress_bar.progress((i + 1) / len(algs_to_check))
            
            status_text.text("Benchmark zakończony!")
            
            st.subheader("Wyścig Algorytmów: Score vs Iteracje")
            fig_bench = go.Figure()
            
            # Tabu Search (linia zmienna)
            ts_data = results.get("tabu_search", {}).get("history", [])
            if ts_data:
                df_ts = pd.DataFrame(ts_data)
                fig_bench.add_trace(go.Scatter(
                    x=df_ts['iteration'], 
                    y=df_ts['score'],
                    mode='lines',
                    name='Tabu Search',
                    line=dict(color='firebrick', width=3)
                ))
            
            # Greedy i DSatur (linie poziome - brak historii)
            x_axis = [0, bench_iter]
            
            if "greedy" in results:
                s = results["greedy"]["score"]
                fig_bench.add_trace(go.Scatter(
                    x=x_axis, y=[s, s],
                    mode='lines',
                    name=f'Greedy (Score: {s:.1f})',
                    line=dict(color='royalblue', width=2, dash='dash')
                ))
            
            if "dsatur" in results:
                s = results["dsatur"]["score"]
                fig_bench.add_trace(go.Scatter(
                    x=x_axis, y=[s, s],
                    mode='lines',
                    name=f'DSatur (Score: {s:.1f})',
                    line=dict(color='forestgreen', width=2, dash='dash')
                ))
            
            fig_bench.add_hline(
                y=0, line_dash="dot", line_color="gray",
                annotation_text="Score = 0"
            )
            
            fig_bench.update_layout(
                xaxis_title="Iteracje",
                yaxis_title="Score (Niższy = Lepszy)",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_bench, use_container_width=True)
            
            st.subheader("Podsumowanie Wydajności")
            summary_data = []
            
            for k, v in results.items():
                summary_data.append({
                    "Algorytm": k.upper(),
                    "Score": v['score'],
                    "Czas [ms]": v['time_seconds'] * 1000,
                    "Konflikty": v['conflicts'],
                    "Poprawa vs Greedy": 0  # Placeholder
                })
            
            df_summary = pd.DataFrame(summary_data)
            
            # Oblicz poprawę względem Greedy
            if "greedy" in results:
                greedy_score = results["greedy"]["score"]
                df_summary["Poprawa vs Greedy"] = df_summary["Score"].apply(
                    lambda x: f"{greedy_score - x:+.1f}"
                )
            
            # Kolorowanie wiersza z najlepszym wynikiem
            def highlight_best(row):
                if row["Konflikty"] == 0:
                    return ['background-color: lightgreen'] * len(row)
                return [''] * len(row)
            
            styled_df = df_summary.style.format({
                "Score": "{:.2f}",
                "Czas [ms]": "{:.2f}"
            }).apply(highlight_best, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Zwycięzca
            best_alg = min(results.items(), key=lambda x: x[1]['score'])
            st.success(
                f"**Zwycięzca**: {best_alg[0].upper()} "
                f"(Score: {best_alg[1]['score']:.2f}, "
                f"Konflikty: {best_alg[1]['conflicts']})"
            )

else:
    st.info("**Skonfiguruj dane w menu po lewej**, aby rozpocząć.")
