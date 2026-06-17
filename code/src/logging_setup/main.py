
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler


def setting_logging(log_path: str, log_level: str):
    # Ensure the ../logs directory exists
    log_directory = Path.cwd() / log_path
    log_directory.mkdir(parents=True, exist_ok=True)

    # Path for the log file
    log_file_path = log_directory / "application.log"

    # Set up TimedRotatingFileHandler to rotate logs every day
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when="midnight",
        interval=1,
        backupCount=0,  # Rotate every midnight, keep all backups
    )

    # Set the file handler's log format
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    match log_level:
        case "DEBUG":
            level = logging.DEBUG
        case "INFO":
            level = logging.INFO
        case "WARNING":
            level = logging.WARNING
        case "ERROR":
            level = logging.ERROR
        case "CRITICAL":
            level = logging.CRITICAL
        case _:
            level = logging.INFO

    # Set up logging configuration
    logging.basicConfig(
        level=level,  # Set to the desired log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Console log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Custom date format
        handlers=[
            file_handler,  # Log to the rotating file in ../logs
            logging.StreamHandler(),  # Log to console
        ],
    )

    # Example logger
    logger = logging.getLogger(__name__)

    # Set the log level for the logger
    logger.setLevel(level)

    # Log a test message to verify
    logger.info("Logging is set up with daily rotation.")

    return logger
