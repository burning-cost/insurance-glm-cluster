import warnings

warnings.warn(
    "insurance-glm-cluster is deprecated. Install insurance-glm-tools instead: pip install insurance-glm-tools",
    DeprecationWarning,
    stacklevel=2,
)

from insurance_glm_tools.cluster import *  # noqa: F401, F403, E402
