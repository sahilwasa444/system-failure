from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemConfiguration:
    users: int
    api_instances: int
    db_connections: int
    cache_enabled: bool

    def __post_init__(self):
        if self.users < 0:
            raise ValueError(f"users must be non-negative, got {self.users}")
        if self.api_instances < 1:
            raise ValueError(f"api_instances must be >= 1, got {self.api_instances}")
        if self.db_connections < 1:
            raise ValueError(f"db_connections must be >= 1, got {self.db_connections}")

    @property
    def total_connections(self) -> int:
        """Total connections across all API instances."""
        return self.api_instances * self.db_connections

    def __repr__(self) -> str:
        return (
            f"SystemConfiguration(users={self.users}, "
            f"api_instances={self.api_instances}, "
            f"db_connections={self.db_connections}, "
            f"cache_enabled={self.cache_enabled})"
        )


@dataclass
class SystemMetrics:
    cpu_usage: float        # 0.0 – 1.0
    memory_usage: float     # 0.0 – 1.0
    latency: float          # milliseconds
    failure: bool
    config: Optional[SystemConfiguration] = field(default=None, repr=False)

    def __post_init__(self):
        if not (0.0 <= self.cpu_usage <= 1.0):
            raise ValueError(f"cpu_usage must be 0.0–1.0, got {self.cpu_usage}")
        if not (0.0 <= self.memory_usage <= 1.0):
            raise ValueError(f"memory_usage must be 0.0–1.0, got {self.memory_usage}")
        if self.latency < 0:
            raise ValueError(f"latency must be non-negative, got {self.latency}")

    @property
    def is_healthy(self) -> bool:
        """True if the system is within safe operating thresholds."""
        return (
            not self.failure
            and self.cpu_usage < 0.9
            and self.memory_usage < 0.9
            and self.latency < 500
        )

    @property
    def status(self) -> str:
        if self.failure:
            return "critical"
        if self.cpu_usage > 0.8 or self.memory_usage > 0.8 or self.latency > 300:
            return "degraded"
        return "healthy"

    def summary(self) -> str:
        return (
            f"[{self.status.upper()}] "
            f"CPU: {self.cpu_usage:.0%} | "
            f"Mem: {self.memory_usage:.0%} | "
            f"Latency: {self.latency:.1f}ms"
        )