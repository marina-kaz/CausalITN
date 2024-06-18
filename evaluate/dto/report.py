from dataclasses import dataclass
from typing import TypedDict

@dataclass
class Metrics:
    wer: float
    cer: float
    accuracy: float


@dataclass
class Sample:
    source: str
    target: str
    prediction: str


@dataclass
class SampleReport:
    global_id: int
    sample: Sample
    metrics: Metrics | None = None


@dataclass
class Parameters:
    temperature: float
    shots: int

    def __repr__(self) -> str:
        return f't{self.temperature}_r{self.shots}'

class MetricsReport(TypedDict):
    global_metrics: Metrics
    detailed_metrics: list[Sample]

# class ModelReport(TypedDict):
