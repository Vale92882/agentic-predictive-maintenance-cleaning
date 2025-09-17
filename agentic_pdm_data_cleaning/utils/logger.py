import logging
import logging.config
import os
import threading

import yaml


class LoggerFactory:

    _lock = threading.Lock()
    _instance = None

    def __new__(cls, config=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LoggerFactory, cls).__new__(cls)
                cls._instance._initialize(config)
            return cls._instance

    def _initialize(self, config):
        if config is None:
            config = self.load_config()
        self.setup_logger(config)

    def get_logger(name=None):

        return logging.getLogger(name)

    def get_logger_with_file(self, benchmark_id, experiment_name, log_file_path):
        """
        Returns a logger with an additional FileHandler writing to file_path.
        Uses the logger's existing level and formatter by default.
        """

        logger = logging.getLogger(f"{benchmark_id}_{experiment_name}")

        # Determine existing logger level
        level = logger.level if logger.level != 0 else logging.DEBUG

        # Retrieve existing formatter from any handler
        formatter = None
        for handler in logger.handlers:
            if handler.formatter:
                formatter = handler.formatter
                break

        # If no formatter found, fallback to default
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Optional: remove existing FileHandlers if you want to replace instead of add
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        # Create and attach the new file handler
        file_handler = logging.FileHandler(
            log_file_path, mode='a')  # use mode 'a' as default
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.debug(
            f"Added FileHandler to logger '{benchmark_id}' with output file '{log_file_path}'")
        return logger

    @staticmethod
    def load_config():
        with open('config/app_config.yaml', 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def setup_logger(config: dict):

        log_config = config['logger']
        log_settings = config['log']

        if not log_settings.get('enabled', True):
            # Disable all logging if 'enabled' is False
            logging.disable(logging.CRITICAL)
            return

        # Create log directory if it does not exist
        log_dir = log_settings.get('directory', './logs')
        os.makedirs(log_dir, exist_ok=True)

        # Read desired handler and formatter from settings, fallback to defaults
        selected_handler = log_settings.get('handler', 'file')
        selected_formatter = log_settings.get('formatter', 'default')

        handlers_available = log_config.get('handlers', {})
        formatters_available = log_config.get('formatters', {})

        # Check handler availability
        if selected_handler not in handlers_available:
            print("Selected handler not defined. Using minimal console logger.")
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger().warning(
                "Logger fallback: minimal console handler configured due to missing handlers.")
            return

        # Build handler config for only the selected handler
        handler = handlers_available[selected_handler]
        handler_conf = {
            'class': handler['class'],
            'level': handler.get('level', log_settings.get('log_level', 'DEBUG')),
            'formatter': selected_formatter,
        }
        if 'filename' in handler:
            handler_conf['filename'] = os.path.join(
                log_dir, handler['filename'])
        if 'maxBytes' in handler:
            handler_conf['maxBytes'] = handler['maxBytes']
        if 'backupCount' in handler:
            handler_conf['backupCount'] = handler['backupCount']

        handlers_config = {selected_handler: handler_conf}

        # Build formatter config for only the selected formatter
        formatters_config = {
            selected_formatter: {
                'format': formatters_available[selected_formatter]['format']
            }
        }

        # Compose final logging configuration dictionary
        logging_config_dict = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters_config,
            'handlers': handlers_config,
            'root': {
                'level': log_settings.get('log_level', 'DEBUG'),
                'handlers': [selected_handler],
            },
        }

        # Apply configuration
        logging.config.dictConfig(logging_config_dict)
        print(
            f"Logger initialized with handler '{selected_handler}' and formatter '{selected_formatter}'")
