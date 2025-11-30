from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- Modele Wejściowe (Request) ---

class Guest(BaseModel):
    id: int
    name: str = "Guest"
    size: int = 1  # Rozmiar grupy (np. para = 2)

class Relationship(BaseModel):
    source: int
    target: int
    weight: int = 0
    is_conflict: bool = False

class SolverParams(BaseModel):
    max_iterations: int = 1000
    tabu_tenure: int = 10

class SolveRequest(BaseModel):
    algorithm: str  # "greedy", "dsatur", "tabu_search"
    num_tables: int
    table_capacity: int
    guests: List[Guest]
    relationships: List[Relationship]
    params: Optional[SolverParams] = None

# --- Modele Wyjściowe (Response) ---

class SolverResponse(BaseModel):
    algorithm: str
    score: float
    conflicts: int
    time_seconds: float
    assignment: Dict[int, int]        # {guest_id: table_id}
    arrangements: Dict[int, List[int]] # {table_id: [guest_order...]} -> DLA OKRĄGŁYCH STOŁÓW
    history: List[Dict[str, Any]]     # Do wykresu konwergencji
