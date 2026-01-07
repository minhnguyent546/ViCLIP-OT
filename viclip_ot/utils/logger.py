"""
Use this class with:

```python
from viclip_ot.utils.logger import logger, init_logger

init_logger()  # should be called at the start of your script
logger.info('This is an info message')
```
"""

import sys
from typing import Any

from loguru import logger


def init_logger(
    level: str = "DEBUG",
    log_file: str | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    if compact:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}:{line}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> - "
            "<level>{message}</level>"
        )
    logger.remove()
    stdout_id = logger.add(
        sys.stdout,
        format=fmt,
        level=level,
    )
    log_file_id = None
    if log_file:
        log_file_id = logger.add(
            log_file,
            format=fmt,
            level="DEBUG",
        )
    return {
        "stdout_id": stdout_id,
        "log_file_id": log_file_id,
        "fmt": fmt,
        "level": level,
    }
