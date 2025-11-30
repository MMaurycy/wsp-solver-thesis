from typing import Dict, Tuple, List, Optional, Any
import networkx as nx
import random
import time
from src.solver.objectives import calculate_grouping_score
from src.solver.dsatur import solve_dsatur

# Hiperparametry dla Tabu Search
MAX_ITERATIONS = 1000
TABU_TENURE = 10
MAX_NO_IMPROVEMENT = 50

def generate_initial_assignment(graph: nx.Graph, num_tables: int) -> Dict[int, int]:
    assignment = {}
    guests = list(graph.nodes())
    random.shuffle(guests)
    for i, guest_id in enumerate(guests):
        assignment[guest_id] = (i % num_tables) + 1
    return assignment

def get_possible_moves(
    current_assignment: Dict[int, int], 
    num_tables: int, 
    tabu_list: List[Tuple[int, int, int]]
) -> List[Tuple[int, int, int]]:
    moves = []
    for guest_id, old_table in current_assignment.items():
        for new_table in range(1, num_tables + 1):
            if new_table != old_table:
                move = (guest_id, old_table, new_table)
                is_tabu = (guest_id, new_table, old_table) in tabu_list
                if not is_tabu:
                    moves.append(move)
    return moves

def evaluate_move(graph, current_assignment, move, num_tables):
    guest_id, _, new_table = move
    temp = current_assignment.copy()
    temp[guest_id] = new_table
    score, _ = calculate_grouping_score(graph, temp, num_tables)
    return float(score)

def solve_tabu(
    graph: nx.Graph, 
    num_tables: int,
    table_capacity: int,  # Dodane, żeby pasowało do sygnatury w core.py
    max_iterations: int = 1000,
    tabu_tenure: int = 10,
    initial_assignment: Optional[Dict[int, int]] = None
) -> Tuple[Dict[int, int], float, List[Dict[str, Any]], float]:
    
    start_time = time.time()
    
    # 1. Startujemy z DSatur jako warm-start
    if not initial_assignment:
        current_assignment, current_score, _ = solve_dsatur(graph, num_tables, table_capacity)
    else:
        current_assignment = initial_assignment
        current_score, _ = calculate_grouping_score(graph, current_assignment, num_tables)

    current_score = float(current_score)
    best_assignment = current_assignment.copy()
    best_score = current_score
    
    tabu_list: List[Tuple[int, int, int]] = []
    history: List[Dict[str, Any]] = []
    no_improvement_count = 0
    
    history.append({'iteration': 0, 'score': current_score, 'best_global_score': best_score, 'conflicts': 0})

    for iteration in range(max_iterations):
        if no_improvement_count >= MAX_NO_IMPROVEMENT:
            # Dywersyfikacja: jeśli utknęliśmy, przetasujmy trochę
            # (Tu można dodać logikę dywersyfikacji, na razie break)
            pass 
            
        moves = get_possible_moves(current_assignment, num_tables, tabu_list)
        if not moves: break
        
        best_local_move = None
        best_local_score = float('inf')
        
        # Ograniczamy przeszukiwanie sąsiedztwa dla wydajności (losowe 50 ruchów)
        random.shuffle(moves)
        sample_moves = moves[:50] if len(moves) > 50 else moves

        for move in sample_moves:
            new_score = evaluate_move(graph, current_assignment, move, num_tables)
            if new_score < best_local_score:
                best_local_score = new_score
                best_local_move = move
        
        if best_local_move:
            guest_id, old_table, new_table = best_local_move
            current_assignment[guest_id] = new_table
            
            if best_local_score < current_score:
                current_score = best_local_score
                no_improvement_count = 0
            else:
                current_score = best_local_score
                no_improvement_count += 1
            
            if current_score < best_score:
                best_score = current_score
                best_assignment = current_assignment.copy()
            
            tabu_list.append((guest_id, old_table, new_table))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        
        if iteration % 50 == 0:
            print(f"TS Iter {iteration}: Score={current_score}")

        history.append({
            'iteration': iteration + 1, 
            'score': current_score, 
            'best_global_score': best_score, 
            'conflicts': 0 # Uproszczenie dla szybkości
        })

    end_time = time.time()
    return best_assignment, best_score, history, (end_time - start_time)
