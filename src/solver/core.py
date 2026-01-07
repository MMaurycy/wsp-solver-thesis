import networkx as nx
import time
from typing import Dict, Any

# Importujemy nasze moduły
from src.solver.greedy import solve_greedy
from src.solver.dsatur import solve_dsatur
from src.solver.tabu_search import solve_tabu  # <--- POPRAWIONY IMPORT
from src.solver.tsp_solver import optimize_all_tables
from src.solver.objectives import calculate_grouping_score

def run_solver(
    algorithm: str,
    graph: nx.Graph,
    num_tables: int,
    table_capacity: int,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    
    if params is None: params = {}
    start_time = time.time()
    print(f"Uruchamiam Etap 1: {algorithm.upper()}")
    
    history = []
    
    if algorithm == "greedy":
        assignment, score, conflicts = solve_greedy(graph, num_tables, table_capacity)
        history = [{'iteration': 0, 'score': score, 'conflicts': conflicts}]
        
    elif algorithm == "dsatur":
        assignment, score, conflicts = solve_dsatur(graph, num_tables, table_capacity)
        history = [{'iteration': 0, 'score': score, 'conflicts': conflicts}]
        
    elif algorithm == "tabu_search":
        iterations = params.get("max_iterations", 1000)
        tenure = params.get("tabu_tenure", 10)
        
        # Wywołanie poprawionej funkcji
        assignment, score, history, _ = solve_tabu(
            graph, num_tables, table_capacity, 
            max_iterations=iterations, 
            tabu_tenure=tenure
        )
        
    else:
        raise ValueError(f"Nieznany algorytm: {algorithm}")


    print("Uruchamiam Etap 2: Optymalizacja kolejności (TSP)")
    arrangements = optimize_all_tables(graph, assignment, num_tables)
    
    total_time = time.time() - start_time
    final_score, final_conflicts = calculate_grouping_score(graph, assignment, num_tables)
    
    return {
        "algorithm": algorithm,
        "score": final_score,
        "conflicts": final_conflicts,
        "time_seconds": total_time,
        "assignment": assignment,
        "arrangements": arrangements,
        "history": history
    }
