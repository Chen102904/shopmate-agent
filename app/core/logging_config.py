import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = BASE_DIR / "logs"

LOG_FILE_NAME = "shopmate.log"
MAX_LOG_SIZE = 10 * 1024 * 1024
BACKUP_COUNT = 5


class RequestIdFilter(logging.Filter):
    """自动为日志记录添加当前请求的 request_id。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from app.core.request_context import get_request_id

            record.request_id = get_request_id()
        except Exception:
            record.request_id = "-"

        return True


def configure_logging() -> None:
    """初始化 ShopMate 统一日志配置。"""

    root_logger = logging.getLogger()

    log_level = os.getenv(
        "SHOPMATE_LOG_LEVEL",
        DEFAULT_LOG_LEVEL,
    ).upper()

    level = getattr(
        logging,
        log_level,
        logging.INFO,
    )

    log_dir = Path(
        os.getenv(
            "SHOPMATE_LOG_DIR",
            str(DEFAULT_LOG_DIR),
        )
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_dir / LOG_FILE_NAME

    root_logger.setLevel(level)

    # 清理之前由本模块创建的 Handler，避免重复输出。
    for handler in root_logger.handlers[:]:
        if getattr(handler, "_shopmate_managed", False):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | request_id=%(request_id)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    request_id_filter = RequestIdFilter()

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)

    # 文件日志：达到10MB自动轮转，保留5个备份。
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_id_filter)

    # 标记自己创建的 Handler，方便重复初始化时清理。
    console_handler._shopmate_managed = True
    file_handler._shopmate_managed = True

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger.info(
        "ShopMate 日志系统初始化完成 | log_file=%s | level=%s",
        log_file,
        log_level,
    )