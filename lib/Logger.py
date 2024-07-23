import logging
import datetime
import os

def setup_logging(verbosity: str, output_dir: str) -> None:
    """Sets up a basic logging configuration.

    Args:
        verbosity: The logging level. Could be: 'debug', 'info', 'warning', 'error', 'critical'.
        output_dir: The output directory of the folder. The log folder will be created under this directory.
    """

    verbosity_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }

    level = verbosity_map.get(verbosity.lower(), logging.INFO)  # Default to INFO level if unknown level passed.

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("{}/log".format(output_dir), exist_ok=True)
    log_filename = "{}/log/{}.log".format(output_dir, current_time)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    

def log_args(args):
    logging.info("Arguments provided:")
    if not isinstance(args, dict):
        args = vars(args)
    for arg_name, arg_value in args.items():
        logging.info("- {}: {}".format(arg_name, arg_value))
    logging.info("=" * 20)