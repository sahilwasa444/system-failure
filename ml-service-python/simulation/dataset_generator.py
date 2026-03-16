import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from .system_model import SystemConfiguration
from .simulation_engine import SimulationEngine, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Defines the sampling ranges for random system configurations."""
    users_min: int = 100
    users_max: int = 100_000
    api_instances_min: int = 1
    api_instances_max: int = 10
    db_connections_min: int = 10
    db_connections_max: int = 200

    def __post_init__(self):
        pairs = [
            ("users", self.users_min, self.users_max),
            ("api_instances", self.api_instances_min, self.api_instances_max),
            ("db_connections", self.db_connections_min, self.db_connections_max),
        ]
        for name, lo, hi in pairs:
            if lo < 1:
                raise ValueError(f"{name}_min must be >= 1, got {lo}")
            if lo > hi:
                raise ValueError(f"{name}_min ({lo}) must be <= {name}_max ({hi})")


@dataclass
class GenerationSummary:
    """Statistics produced after a generation run."""
    samples: int
    output_file: Path
    failure_count: int
    cache_enabled_count: int

    @property
    def failure_rate(self) -> float:
        return self.failure_count / self.samples if self.samples else 0.0

    @property
    def cache_enabled_rate(self) -> float:
        return self.cache_enabled_count / self.samples if self.samples else 0.0

    def __str__(self) -> str:
        return (
            f"Generated {self.samples:,} samples → {self.output_file}\n"
            f"  Failure rate:      {self.failure_rate:.1%}\n"
            f"  Cache enabled:     {self.cache_enabled_rate:.1%}"
        )


# Column definitions: (header_name, extractor_callable)
_COLUMNS: list[tuple[str, callable]] = [
    ("users",           lambda c, m: c.users),
    ("api_instances",   lambda c, m: c.api_instances),
    ("db_connections",  lambda c, m: c.db_connections),
    ("cache_enabled",   lambda c, m: int(c.cache_enabled)),
    ("cpu_usage",       lambda c, m: round(m.cpu_usage, 6)),
    ("memory_usage",    lambda c, m: round(m.memory_usage, 6)),
    ("latency",         lambda c, m: round(m.latency, 4)),
    ("failure",         lambda c, m: int(m.failure)),
]

HEADER = [col for col, _ in _COLUMNS]


class DatasetGenerator:
    """
    Generates synthetic (config → metrics) datasets for ML training.

    Parameters
    ----------
    output_file : str or Path
        Destination CSV path. Parent directories are created automatically.
    seed : int, optional
        Master seed. Config sampling and simulation use independent RNGs
        derived from this value, so both are reproducible yet uncorrelated.
    gen_config : GeneratorConfig, optional
        Sampling ranges for random system configurations.
    sim_config : SimulationConfig, optional
        Physical model coefficients forwarded to SimulationEngine.
    """

    def __init__(
        self,
        output_file: str | Path = "dataset.csv",
        seed: Optional[int] = 42,
        gen_config: Optional[GeneratorConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
    ):
        self.output_file = Path(output_file)
        self.gen_config = gen_config or GeneratorConfig()
        # Two independent RNGs from the same master seed — config sampling
        # and simulation noise stay reproducible but don't interfere.
        self._rng = random.Random(seed)
        self.engine = SimulationEngine(
            sim_config=sim_config,
            seed=None if seed is None else seed + 1,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, samples: int = 10_000, log_every: int = 1_000
    ) -> GenerationSummary:
        """
        Stream `samples` rows to CSV and return a GenerationSummary.

        Rows are written immediately — no full dataset held in memory.
        """
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        failure_count = 0
        cache_enabled_count = 0

        logger.info("Starting generation: %d samples → %s", samples, self.output_file)

        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

            for i, (config, metrics) in enumerate(self._sample_stream(samples), start=1):
                writer.writerow(self._to_row(config, metrics))
                if metrics.failure:
                    failure_count += 1
                if config.cache_enabled:
                    cache_enabled_count += 1
                if log_every and i % log_every == 0:
                    logger.info("  %d / %d rows written", i, samples)

        summary = GenerationSummary(
            samples=samples,
            output_file=self.output_file,
            failure_count=failure_count,
            cache_enabled_count=cache_enabled_count,
        )
        logger.info(str(summary))
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_stream(
        self, samples: int
    ) -> Iterator[tuple[SystemConfiguration, object]]:
        """Yield (config, metrics) pairs one at a time."""
        for _ in range(samples):
            config = self._random_config()
            yield config, self.engine.simulate(config)

    def _random_config(self) -> SystemConfiguration:
        gc = self.gen_config
        return SystemConfiguration(
            users=self._rng.randint(gc.users_min, gc.users_max),
            api_instances=self._rng.randint(gc.api_instances_min, gc.api_instances_max),
            db_connections=self._rng.randint(gc.db_connections_min, gc.db_connections_max),
            cache_enabled=self._rng.choice([True, False]),
        )

    @staticmethod
    def _to_row(config: SystemConfiguration, metrics) -> list:
        return [extractor(config, metrics) for _, extractor in _COLUMNS]