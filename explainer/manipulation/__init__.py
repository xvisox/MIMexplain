from explainer.manipulation.engine import (
    DistributionAnchor,
    ManipulationConfig,
    ManipulationProgressSnapshot,
    ManipulationResult,
    MetaClip2ManipulationEngine,
)
from explainer.manipulation.generator import (
    GeneratorConfig,
    PerturbationGenerator,
    apply_generator_delta,
    count_parameters,
    supervised_delta_loss,
)
from explainer.manipulation.perceptual import (
    PerceptualMaskConfig,
    build_perceptual_allowance_map,
    perceptual_budget_map,
)
from explainer.manipulation.universal import (
    UniversalManipulationEngine,
    UniversalManipulationResult,
    apply_universal_delta,
    average_metrics,
    build_uniform_budget,
)

__all__ = [
    "DistributionAnchor",
    "GeneratorConfig",
    "ManipulationConfig",
    "ManipulationProgressSnapshot",
    "ManipulationResult",
    "MetaClip2ManipulationEngine",
    "PerceptualMaskConfig",
    "PerturbationGenerator",
    "UniversalManipulationEngine",
    "UniversalManipulationResult",
    "apply_generator_delta",
    "apply_universal_delta",
    "average_metrics",
    "build_perceptual_allowance_map",
    "build_uniform_budget",
    "count_parameters",
    "perceptual_budget_map",
    "supervised_delta_loss",
]
