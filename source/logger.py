import os
import json
import sys
from datetime import datetime
from typing import Any, TextIO


class Logger(object):
    """
    Enhance logging class, supporting:
        - Structured JSON metrics recording
        - Separate detailed logs and concise metrics
        - Compatible with both console and file output
    """
    def __init__(self, logfile: str | None, metrics_file: str | None = None, *, enabled: bool = True) -> None:
        """Initialize logger

        Args:
            logfile (str): the text log file path (append write)
            metrics_file (str): the metrics JSONL file path (optional, will be cleared when initialized)
        """
        self._enabled = bool(enabled)
        self.log: TextIO | None = None
        self._orig_stdout: TextIO = sys.stdout
        self._closed: bool = False

        if self._enabled and logfile is not None:
            logfile_path = logfile
            self.log = open(logfile_path, "a", encoding="utf-8")

        self.metrics_file = metrics_file if self._enabled else None
        if self.metrics_file:
            metrics_path = self.metrics_file
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write("")  # Clear file

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Record structured metrics to JSONL file (one JSON per line)."""
        if not self.metrics_file:
            return

        payload = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write('\n')

    def flush(self) -> None:
        """Flush buffer."""
        try:
            if self.log and not self.log.closed:
                self.log.flush()
        except (OSError, ValueError):
            pass

    def write(self, message: str) -> int:
        """Write text log, return written character count, return 0 if failed."""
        if not self._enabled:
            return 0
        target = self.log if (self.log and not self.log.closed) else self._orig_stdout
        try:
            written = target.write(message)
            return int(written) if isinstance(written, int) else len(message)
        except (OSError, ValueError, RuntimeError, UnicodeEncodeError):
            return 0

    def writelines(self, lines: list[str] | tuple[str, ...]) -> int:
        """Write multiple lines, return accumulated written character count."""
        total = 0
        for line in lines:
            total += self.write(line)
        return total

    def log_config(self, config: dict[str, Any]) -> None:
        """Record configuration to metrics file (first record, one JSON per line)."""
        if not self.metrics_file:
            return

        config_entry = {
            'type': 'config',
            'timestamp': datetime.now().isoformat(),
            **config
        }

        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            json.dump(config_entry, f, ensure_ascii=False)
            f.write('\n')

    @property
    def closed(self) -> bool:
        """Check if the log is closed."""
        if not self._enabled:
            return True
        return self._closed or (self.log is None or getattr(self.log, 'closed', True))

    def __enter__(self) -> "Logger":
        """Enter the context."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit the context."""
        self.close()

    def close(self) -> None:
        """Close the log."""
        if self._closed:
            return
        try:
            if self.log and not self.log.closed:
                self.log.close()
        finally:
            try:
                # only restore when stdout points to self
                if sys.stdout is self:
                    sys.stdout = self._orig_stdout
            except (RuntimeError, AttributeError):
                pass
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, RuntimeError):
            pass


def setup_logger(
    log_root: str,
    dataset_name: str,
    seed: int,
    model_name: str,
    split_type: str,
    *,
    enabled: bool = True
) -> dict[str, object]:
    """
    Setup logger for training with structured logging.

    Args:
        log_root (str): Root directory for logs
        dataset_name (str): Name of the dataset
        seed (int): Random seed
        model_name (str): Name of the model

    Returns:
        dict[str, object]: Validation/test CSV paths plus the logger instance.
    """
    # Build log directory path
    log_dir = os.path.join(log_root, f"{dataset_name}-{seed}-{split_type}", model_name)
    if enabled:
        os.makedirs(log_dir, exist_ok=True)

    # Create log file paths
    log_file = os.path.join(log_dir, f"{model_name}.log")
    metrics_file = os.path.join(log_dir, f"{model_name}_metrics.jsonl")

    # Create logger instance with metrics support
    logger = Logger(
        logfile=log_file if enabled else None,
        metrics_file=metrics_file if enabled else None,
        enabled=enabled,
    )

    # Redirect stdout to logger
    if enabled:
        sys.stdout = logger

    # Log initial message
    logger.write(f"=== Training started at {datetime.now().isoformat()} ===\n")
    logger.write(f"Model: {model_name}\n")
    logger.write(f"Dataset: {dataset_name}\n")
    logger.write(f"Seed: {seed}\n")
    logger.write(f"Log directory: {log_dir}\n")
    logger.write("=" * 50 + "\n")
    logger.flush()

    # Return dictionary with paths and logger
    return {
        "validation_results_csv": os.path.join(log_dir, "validation_results.csv"),
        "test_results_csv": os.path.join(log_dir, "test_results.csv"),
        "logger": logger,
    }
