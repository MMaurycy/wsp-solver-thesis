from fastapi import FastAPI, HTTPException
import networkx as nx
from src.models import SolveRequest, SolverResponse
from src.solver.core import run_solver

app = FastAPI(title="WSP Solver API")

def build_graph(guests, relationships) -> nx.Graph:
    """
    Tworzy graf NetworkX na podstawie danych wejściowych.
    Zawiera AGRESYWNĄ konwersję typów, aby uniknąć błędów str/int.
    """
    G = nx.Graph()
    
    # 1. Dodaj wierzchołki (Gości)
    for g in guests:
        # Wymuszamy konwersję na int
        try:
            g_id = int(g.id)
            g_size = int(g.size)
        except ValueError:
            g_id = g.id
            g_size = 1
            
        G.add_node(g_id, size=g_size, name=str(g.name))
        
    # 2. Dodaj krawędzie (Relacje)
    for r in relationships:
        try:
            src = int(r.source)
            tgt = int(r.target)
            # Kluczowa poprawka: waga musi być intem
            w = int(r.weight)
        except ValueError:
            continue

        G.add_edge(
            src, 
            tgt, 
            weight=w, 
            conflict=bool(r.is_conflict)
        )
    return G

@app.get("/")
def health_check():
    return {"status": "ok", "message": "WSP Solver API is running"}

@app.post("/solve", response_model=SolverResponse)
def solve_endpoint(request: SolveRequest):
    try:
        # 1. Budowa modelu grafowego (z czyszczeniem typów)
        graph = build_graph(request.guests, request.relationships)
        
        # 2. Przygotowanie parametrów
        params_dict = {}
        if request.params:
            params_dict = request.params.dict()

        # 3. Uruchomienie Solvera
        result = run_solver(
            algorithm=request.algorithm,
            graph=graph,
            num_tables=request.num_tables,
            table_capacity=request.table_capacity,
            params=params_dict
        )
        
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
