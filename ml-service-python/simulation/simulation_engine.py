import random
from dataclasses import dataclass, field
from typing import Optional
from .system_model import SystemConfiguration, SystemMetrics


@dataclass
class SimulationConfig:
    """Tunable knobs for the simulation model."""
    # CPU model
    cpu_load_scale: float = 20_000   # users per server at 100% CPU
    cpu_noise_min: float = 0.05
    cpu_noise_max: float = 0.15

    # Memory model
    memory_user_scale: float = 100_000  # users at 100% memory
    memory_noise_min: float = 0.05
    memory_noise_max: float = 0.20

    # Latency model
    latency_load_scale: float = 50    # load units per ms of base latency
    latency_cache_factor: float = 0.7 # cache reduces latency by 30%
    latency_noise_min: float = 10.0   # ms
    latency_noise_max: float = 40.0   # ms

    # Failure thresholds
    cpu_failure_threshold: float = 0.95
    memory_failure_threshold: float = 0.95
    latency_failure_threshold: float = 800.0  # ms

    def __post_init__(self):
        for name in ("cpu_noise_min", "cpu_noise_max",
                     "memory_noise_min", "memory_noise_max"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be 0.0–1.0, got {v}")
        if self.latency_noise_min > self.latency_noise_max:
            raise ValueError("latency_noise_min must be <= latency_noise_max")


class SimulationEngine:
    """
    Runs a stochastic performance simulation against a SystemConfiguration.

    Parameters
    ----------
    sim_config : SimulationConfig, optional
        Model coefficients and failure thresholds. Defaults to production values.
    seed : int, optional
        Fix for reproducible results (testing, benchmarking).
    """

    def __init__(
        self,
        sim_config: Optional[SimulationConfig] = None,
        seed: Optional[int] = None,
    ):
        self.sim_config = sim_config or SimulationConfig()
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, config: SystemConfiguration) -> SystemMetrics:
        """Run a single simulation pass and return observed metrics."""
        load_per_server = self._load_per_server(config)
        cpu = self._cpu_usage(load_per_server)
        mem = self._memory_usage(config.users)
        lat = self._latency(load_per_server, config.cache_enabled)
        failure = self._has_failure(cpu, mem, lat)
        return SystemMetrics(
            cpu_usage=cpu,
            memory_usage=mem,
            latency=lat,
            failure=failure,
            config=config,
        )

    def simulate_many(
        self, config: SystemConfiguration, runs: int = 100
    ) -> list[SystemMetrics]:
        """Run `runs` independent simulations and return all results."""
        if runs < 1:
            raise ValueError(f"runs must be >= 1, got {runs}")
        return [self.simulate(config) for _ in range(runs)]

    # ------------------------------------------------------------------
    # Private helpers — each models one physical quantity
    # ------------------------------------------------------------------

    def _load_per_server(self, config: SystemConfiguration) -> float:
        return config.users / config.api_instances

    def _cpu_usage(self, load_per_server: float) -> float:
        sc = self.sim_config
        deterministic = load_per_server / sc.cpu_load_scale
        noise = self._rng.uniform(sc.cpu_noise_min, sc.cpu_noise_max)
        return min(1.0, deterministic + noise)

    def _memory_usage(self, users: int) -> float:
        sc = self.sim_config
        deterministic = users / sc.memory_user_scale
        noise = self._rng.uniform(sc.memory_noise_min, sc.memory_noise_max)
        return min(1.0, deterministic + noise)

    def _latency(self, load_per_server: float, cache_enabled: bool) -> float:
        sc = self.sim_config
        base = load_per_server / sc.latency_load_scale
        if cache_enabled:
            base *= sc.latency_cache_factor
        noise = self._rng.uniform(sc.latency_noise_min, sc.latency_noise_max)
        return base + noise

    def _has_failure(
        self, cpu_usage: float, memory_usage: float, latency: float
    ) -> bool:
        sc = self.sim_config
        return (
            cpu_usage > sc.cpu_failure_threshold
            or memory_usage > sc.memory_failure_threshold
            or latency > sc.latency_failure_threshold
        )