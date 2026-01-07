from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from enum import Enum

class AlgorithmType(str, Enum):
    GREEDY = "greedy"
    DSATUR = "dsatur"
    TABU_SEARCH = "tabu_search"

class ScenarioType(str, Enum):
    BALANCED = "balanced"
    CONFLICTED = "conflicted"
    PEACEFUL = "peaceful"

class Guest(BaseModel):
    id: int
    name: str = "Guest"
    size: int = Field(1, ge=1, le=6, description="Rozmiar grupy (1-6 osób)")

class Relationship(BaseModel):
    source: int
    target: int
    weight: int = Field(0, description="Waga relacji (+1000=konflikt, -5=preferencja)")
    is_conflict: bool = False

class SolverParams(BaseModel):
    max_iterations: int = Field(1000, ge=100, le=10000)
    tabu_tenure: int = Field(10, ge=5, le=50)
    beta: float = Field(1.0, ge=0.0, le=10.0, description="Współczynnik kary za balans")

class GenerateDataRequest(BaseModel):
    """
    Parametry generowania syntetycznych danych.
    KLUCZOWA ZMIANA: total_people jako główny parametr.
    """
    total_people: int = Field(
        100, 
        ge=20, 
        le=500,
        description="Całkowita liczba osób (gości weselnych)"
    )
    num_groups: Optional[int] = Field(
        None,
        ge=5,
        le=200,
        description="Liczba grup (domyślnie: total_people // 3)"
    )
    table_capacity: int = Field(
        10,
        ge=6,
        le=14,
        description="Pojemność jednego stołu"
    )
    scenario: ScenarioType = Field(
        ScenarioType.BALANCED,
        description="Typ generowanych relacji"
    )
    conflict_probability: float = Field(
        0.15,
        ge=0.0,
        le=0.5,
        description="Prawdopodobieństwo konfliktu między grupami"
    )
    preference_probability: float = Field(
        0.20,
        ge=0.0,
        le=0.5,
        description="Prawdopodobieństwo preferencji między grupami"
    )

    @validator('num_groups', always=True)
    def set_default_groups(cls, v, values):
        if v is None and 'total_people' in values:
            return max(10, values['total_people'] // 3)
        return v
    
    @validator('num_groups')
    def validate_groups_vs_people(cls, v, values):
        if 'total_people' in values and v > values['total_people']:
            raise ValueError(
                f"Liczba grup ({v}) nie może być większa niż liczba osób ({values['total_people']})"
            )
        return v

class GenerateDataResponse(BaseModel):
    guests: List[Guest]
    relationships: List[Relationship]
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statystyki wygenerowanych danych"
    )

class SolveRequest(BaseModel):
    algorithm: AlgorithmType
    num_tables: int = Field(ge=2, le=50)
    table_capacity: int = Field(ge=4, le=20)
    guests: List[Guest]
    relationships: List[Relationship]
    params: Optional[SolverParams] = Field(default_factory=SolverParams)

    @validator('num_tables')
    def validate_capacity(cls, v, values):
        if 'guests' in values and 'table_capacity' in values:
            total_people = sum(g.size for g in values['guests'])
            total_capacity = v * values['table_capacity']
            if total_people > total_capacity:
                raise ValueError(
                    f"Za mało miejsc! Goście: {total_people}, "
                    f"Pojemność: {total_capacity} (stoły: {v} × {values['table_capacity']})"
                )
        return v

class SolverResponse(BaseModel):
    algorithm: str
    score: float
    conflicts: int
    time_seconds: float
    assignment: Dict[int, int]         # {guest_id: table_id}
    arrangements: Dict[int, List[int]]  # {table_id: [guest_order...]}
    history: List[Dict[str, Any]]      # Historia konwergencji
    metadata: Dict[str, Any] = Field(  # NOWE: dodatkowe statystyki
        default_factory=dict,
        description="Statystyki rozwiązania (balans, wykorzystanie stołów, itp.)"
    )

class ProblemStats(BaseModel):
    """Statystyki instancji problemu (do wyświetlenia w UI)."""
    total_people: int
    num_groups: int
    num_tables: int
    total_capacity: int
    avg_group_size: float
    num_conflicts: int
    num_preferences: int
    utilization_percent: float  # total_people / total_capacity * 100
