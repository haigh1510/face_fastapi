import logging


def get_logger():
    logger_name = 'FACE_FASTAPI_LOGGER'

    if logger_name in logging.Logger.manager.loggerDict.keys():
        return logging.getLogger(logger_name)

    logger = logging.getLogger(logger_name)

    c_format = logging.Formatter(
        fmt='%(asctime)s %(name)s [%(levelname)s] - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    c_handler = logging.StreamHandler()
    c_handler.setFormatter(c_format)

    logger.addHandler(c_handler)
    logger.setLevel(logging.DEBUG)

    logger.debug(f'{logger_name} initialized')

    return logger
