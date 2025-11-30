import networkx as nx
from typing import Dict, List, Tuple
from src.solver.objectives import calculate_grouping_score

def solve_dsatur(
    graph: nx.Graph, 
    num_tables: int, 
    table_capacity: int
) -> Tuple[Dict[int, int], float, int]:
    """
    Algorytm DSatur (Heurystyka):
    Wybiera wierzchołek o największym stopniu nasycenia.
    """
    assignment: Dict[int, int] = {}
    table_loads: Dict[int, int] = {i: 0 for i in range(1, num_tables + 1)}
    
    unassigned_nodes = list(graph.nodes())
    
    while unassigned_nodes:
        def get_saturation_degree(node):
            used_tables = set()
            for neighbor in graph.neighbors(node):
                if neighbor in assignment:
                    used_tables.add(assignment[neighbor])
            return len(used_tables)
            
        # Sortowanie dynamiczne: Nasycenie -> Stopień -> Rozmiar
        unassigned_nodes.sort(key=lambda n: (
            get_saturation_degree(n),
            graph.degree(n),
            graph.nodes[n].get('size', 1)
        ), reverse=True)
        
        node_to_assign = unassigned_nodes.pop(0)
        group_size = graph.nodes[node_to_assign].get('size', 1)
        
        best_table = -1
        assigned = False
        
        # Szukamy pierwszego legalnego stołu (First Fit)
        for table_id in range(1, num_tables + 1):
            if table_loads[table_id] + group_size <= table_capacity:
                is_conflict = False
                for neighbor in graph.neighbors(node_to_assign):
                    if assignment.get(neighbor) == table_id:
                        if graph[node_to_assign][neighbor].get('conflict'):
                            is_conflict = True
                            break
                if not is_conflict:
                    assignment[node_to_assign] = table_id
                    table_loads[table_id] += group_size
                    assigned = True
                    break
        
        if not assigned:
             # Fallback
             target_table = min(table_loads, key=table_loads.get)
             assignment[node_to_assign] = target_table
             table_loads[target_table] += group_size

    final_score, conflicts = calculate_grouping_score(graph, assignment, num_tables)
    return assignment, final_score, conflicts
