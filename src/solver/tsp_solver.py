import networkx as nx
import random
from typing import List, Dict
from src.solver.objectives import calculate_arrangement_score

def solve_tsp_2opt(graph: nx.Graph, guests_at_table: List[int]) -> List[int]:
    if len(guests_at_table) <= 2:
        return guests_at_table
    
    current_route = guests_at_table.copy()
    best_score = calculate_arrangement_score(graph, current_route)
    improved = True
    
    while improved:
        improved = False
        n = len(current_route)
        
        for i in range(n - 1):
            for k in range(i + 1, n):
                # Odwracamy segment trasy między i a k
                new_route = current_route[:i] + current_route[i:k+1][::-1] + current_route[k+1:]
                new_score = calculate_arrangement_score(graph, new_route)
                
                if new_score < best_score:
                    current_route = new_route
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
                
    return current_route

def optimize_all_tables(
    graph: nx.Graph, 
    assignment: Dict[int, int], 
    num_tables: int
) -> Dict[int, List[int]]:
    # 1. Pogrupuj gości wg stołów
    tables = {i: [] for i in range(1, num_tables + 1)}
    for guest, table_id in assignment.items():
        tables[table_id].append(guest)
        
    final_arrangements = {}
    
    # 2. Dla każdego stołu uruchom 2-OPT
    for table_id, guests in tables.items():
        if not guests:
            final_arrangements[table_id] = []
            continue
            
        optimized_order = solve_tsp_2opt(graph, guests)
        final_arrangements[table_id] = optimized_order
        
    return final_arrangements
