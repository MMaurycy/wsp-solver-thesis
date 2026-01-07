from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
import random
import math
from typing import List, Dict
from src.models import (
    SolveRequest, 
    SolverResponse, 
    GenerateDataRequest,
    GenerateDataResponse,
    Guest,
    Relationship,
    ProblemStats
)
from src.solver.core import run_solver

app = FastAPI(
    title="WSP Solver API",
    description="API for Wedding Seating Problem optimization",
    version="2.0.0"
)

# CORS (jeśli używasz Streamlit na innym porcie)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
#  HELPER FUNCTIONS (Nowe funkcje pomocnicze)
# ============================================================================

def _distribute_people(total_people: int, num_groups: int) -> List[int]:
    if total_people < num_groups:
        raise ValueError(
            f"Niemożliwe do rozdzielenia: {total_people} osób na {num_groups} grup "
            "(każda grupa musi mieć minimum 1 osobę)"
        )
    
    # Krok 1: Każda grupa dostaje 1 osobę
    sizes = [1] * num_groups
    remaining = total_people - num_groups
    
    # Krok 2: Rozdaj pozostałe osoby
    max_group_size = 6  # Limit z pracy (małżeństwo + dzieci)
    
    while remaining > 0:
        # Losuj grupę
        idx = random.randint(0, num_groups - 1)
        
        # Sprawdź, czy można dodać osobę
        if sizes[idx] < max_group_size:
            sizes[idx] += 1
            remaining -= 1
        # Jeśli wszystkie grupy są pełne (6 osób), zwiększ limit
        elif all(s >= max_group_size for s in sizes):
            sizes[idx] += 1
            remaining -= 1
    
    # Shuffle, żeby rozmiary były w losowej kolejności
    random.shuffle(sizes)
    
    return sizes


def _generate_relationships(
    guests: List[Guest], 
    scenario: str,
    conflict_prob: float = 0.15,
    preference_prob: float = 0.20
) -> List[Relationship]:
    relationships = []
    n = len(guests)
    
    # Dostosuj prawdopodobieństwa do scenariusza
    if scenario == "conflicted":
        conflict_prob *= 2.0
        preference_prob *= 0.5
    elif scenario == "peaceful":
        conflict_prob *= 0.3
        preference_prob *= 1.5
    
    # Generuj relacje dla każdej pary
    for i in range(n):
        for j in range(i + 1, n):
            rand = random.random()
            
            # Konflikt (waga +1000)
            if rand < conflict_prob:
                relationships.append(Relationship(
                    source=guests[i].id,
                    target=guests[j].id,
                    weight=1000,
                    is_conflict=True
                ))
            # Preferencja (waga -5)
            elif rand < conflict_prob + preference_prob:
                relationships.append(Relationship(
                    source=guests[i].id,
                    target=guests[j].id,
                    weight=-5,
                    is_conflict=False
                ))
    
    return relationships


def build_graph(guests: List[Guest], relationships: List[Relationship]) -> nx.Graph:
    G = nx.Graph()
    
    # Dodaj wierzchołki
    for g in guests:
        try:
            g_id = int(g.id)
            g_size = int(g.size)
        except (ValueError, TypeError):
            g_id = g.id
            g_size = 1
        
        G.add_node(g_id, size=g_size, name=str(g.name))
    
    # Dodaj krawędzie
    for r in relationships:
        try:
            src = int(r.source)
            tgt = int(r.target)
            w = int(r.weight)
        except (ValueError, TypeError):
            continue
        
        G.add_edge(src, tgt, weight=w, conflict=bool(r.is_conflict))
    
    return G


# ============================================================================
#  ENDPOINTY API
# ============================================================================

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "WSP Solver API v2.0 is running",
        "endpoints": ["/generate", "/solve", "/stats"]
    }


@app.post("/generate", response_model=GenerateDataResponse)
def generate_test_data(request: GenerateDataRequest):
    try:
        # 1. Oblicz liczbę grup (jeśli nie podano)
        num_groups = request.num_groups or max(10, request.total_people // 3)
        
        # 2. Rozdziel osoby na grupy
        group_sizes = _distribute_people(request.total_people, num_groups)
        
        # 3. Stwórz obiekty gości
        guests = []
        for i, size in enumerate(group_sizes):
            guests.append(Guest(
                id=i,
                name=f"Grupa {i+1}",
                size=size
            ))
        
        # 4. Wygeneruj relacje
        relationships = _generate_relationships(
            guests,
            request.scenario.value,
            request.conflict_probability,
            request.preference_probability
        )
        
        # 5. Statystyki (do wyświetlenia w UI)
        metadata = {
            "total_people": request.total_people,
            "num_groups": len(guests),
            "avg_group_size": request.total_people / len(guests),
            "num_conflicts": sum(1 for r in relationships if r.is_conflict),
            "num_preferences": sum(1 for r in relationships if not r.is_conflict and r.weight < 0),
            "scenario": request.scenario.value,
            "recommended_tables": math.ceil(request.total_people / request.table_capacity)
        }
        
        return GenerateDataResponse(
            guests=guests,
            relationships=relationships,
            metadata=metadata
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/solve", response_model=SolverResponse)
def solve_endpoint(request: SolveRequest):
    try:
        # 1. Walidacja wejścia
        total_people = sum(g.size for g in request.guests)
        total_capacity = request.num_tables * request.table_capacity
        
        if total_people > total_capacity:
            raise ValueError(
                f"Za mało miejsc! Goście: {total_people}, "
                f"Pojemność: {total_capacity}"
            )
        
        # 2. Budowa grafu
        graph = build_graph(request.guests, request.relationships)
        
        # 3. Przygotowanie parametrów
        params_dict = request.params.dict() if request.params else {}
        
        # 4. Uruchomienie solvera
        result = run_solver(
            algorithm=request.algorithm.value,
            graph=graph,
            num_tables=request.num_tables,
            table_capacity=request.table_capacity,
            params=params_dict
        )
        
        # 5. Dodaj metadane do wyniku
        result["metadata"] = {
            "total_people": total_people,
            "num_groups": len(request.guests),
            "utilization_percent": (total_people / total_capacity) * 100
        }
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")


@app.post("/stats", response_model=ProblemStats)
def calculate_stats(request: SolveRequest):
    total_people = sum(g.size for g in request.guests)
    total_capacity = request.num_tables * request.table_capacity
    num_conflicts = sum(1 for r in request.relationships if r.is_conflict)
    num_preferences = sum(
        1 for r in request.relationships 
        if not r.is_conflict and r.weight < 0
    )
    
    return ProblemStats(
        total_people=total_people,
        num_groups=len(request.guests),
        num_tables=request.num_tables,
        total_capacity=total_capacity,
        avg_group_size=total_people / len(request.guests) if request.guests else 0,
        num_conflicts=num_conflicts,
        num_preferences=num_preferences,
        utilization_percent=(total_people / total_capacity * 100) if total_capacity > 0 else 0
    )
