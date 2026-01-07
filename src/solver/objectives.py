import math
import networkx as nx
from typing import Dict, List, Tuple

WEIGHT_CONFLICT = 1000
WEIGHT_PREFERENCE = -5
DEFAULT_BALANCE_WEIGHT = 1.0

def get_relationship_weight(graph: nx.Graph, u: int, v: int) -> int:
    if not graph.has_edge(u, v):
        return 0
    edge_data = graph.get_edge_data(u, v)
    
    if edge_data.get("conflict", False):
        return int(WEIGHT_CONFLICT)
    
    # rzutowanie na int
    val = edge_data.get("weight", WEIGHT_PREFERENCE)
    try:
        return int(val)
    except:
        return int(WEIGHT_PREFERENCE)

def calculate_grouping_score(
    graph: nx.Graph, 
    assignment: Dict[int, int], 
    num_tables: int, 
    balance_penalty_weight: float = DEFAULT_BALANCE_WEIGHT
) -> Tuple[float, int]:
    score = 0.0
    conflicts = 0
    tables: Dict[int, List[int]] = {i: [] for i in range(1, num_tables + 1)}
    table_loads: Dict[int, int] = {i: 0 for i in range(1, num_tables + 1)}
    
    for guest_id, table_id in assignment.items():
        if table_id not in tables: continue
        tables[table_id].append(guest_id)
        # Zabezpieczenie rozmiaru
        sz = graph.nodes[guest_id].get("size", 1)
        table_loads[table_id] += int(sz)

    for table_id, guests in tables.items():
        if len(guests) < 2: continue
        for i in range(len(guests)):
            for j in range(i + 1, len(guests)):
                u, v = guests[i], guests[j]
                weight = get_relationship_weight(graph, u, v)
                
                score += float(weight)
                if weight >= WEIGHT_CONFLICT:
                    conflicts += 1

    active_loads = [float(l) for l in table_loads.values()]
    if len(active_loads) > 0:
        avg_load = sum(active_loads) / len(active_loads)
        variance = sum((l - avg_load) ** 2 for l in active_loads) / len(active_loads)
        std_dev = math.sqrt(variance)
        score += (std_dev * float(balance_penalty_weight))

    return float(score), conflicts

def calculate_arrangement_score(graph: nx.Graph, sequence: List[int]) -> float:
    if len(sequence) < 2: return 0.0
    cost = 0.0
    n = len(sequence)
    for i in range(n):
        u = sequence[i]
        v = sequence[(i + 1) % n]
        weight = get_relationship_weight(graph, u, v)
        cost += float(weight)
    return cost
