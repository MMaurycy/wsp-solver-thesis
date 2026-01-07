import networkx as nx
from typing import Dict, List, Tuple
from src.solver.objectives import calculate_grouping_score, WEIGHT_CONFLICT

def solve_greedy(
    graph: nx.Graph, 
    num_tables: int, 
    table_capacity: int
) -> Tuple[Dict[int, int], float, int]:
    sorted_groups = sorted(
        graph.nodes(data=True), 
        key=lambda x: x[1].get('size', 1), 
        reverse=True
    )
    
    assignment: Dict[int, int] = {}
    table_loads: Dict[int, int] = {i: 0 for i in range(1, num_tables + 1)}
    
    for node_id, data in sorted_groups:
        group_size = data.get('size', 1)
        assigned = False
        
        for table_id in range(1, num_tables + 1):
            if table_loads[table_id] + group_size <= table_capacity:
                conflict_found = False
                for other_guest, other_table in assignment.items():
                    if other_table == table_id:
                        if graph.has_edge(node_id, other_guest):
                            edge_data = graph.get_edge_data(node_id, other_guest)
                            if edge_data.get('conflict', False):
                                conflict_found = True
                                break
                
                if not conflict_found:
                    assignment[node_id] = table_id
                    table_loads[table_id] += group_size
                    assigned = True
                    break
        
        if not assigned:
            target_table = min(table_loads, key=table_loads.get)
            assignment[node_id] = target_table
            table_loads[target_table] += group_size

    final_score, conflicts = calculate_grouping_score(graph, assignment, num_tables)
    return assignment, final_score, conflicts
