import os
import warnings


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

warnings.filterwarnings(
    "ignore",
    message=r".*Could not find the number of physical cores.*",
    category=UserWarning,
)
