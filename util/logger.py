import os
import logbook
from logbook import Logger, TimedRotatingFileHandler
from logbook.more import ColorizedStderrHandler

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.00"


def handler_log_formatter(record, handler):
    log = "[{dt}][{level}][{filename}][{func_name}][{lineno}] {msg}".format(
        dt=record.time,
        level=record.level_name,
        filename=os.path.split(record.filename)[-1],
        func_name=record.func_name,
        lineno=record.lineno,
        msg=record.message,
    )
    return log


std_handler = ColorizedStderrHandler(bubble=True)
std_handler.formatter = handler_log_formatter

proj = 'reductor'
cur_path = os.path.abspath(os.path.dirname(__file__))
dir_proj = cur_path[:cur_path.find(proj)] + proj

dir_log = os.path.join(dir_proj, 'log')
if not os.path.exists(dir_log):
    os.makedirs(dir_log)

file_handler = TimedRotatingFileHandler(
    os.path.join(dir_log, '%s.log' % 'runtime'),
    date_format='%Y%m%d',
    bubble=True,
    backup_count=31,
    rollover_format='{basename}.{timestamp}{ext}')
file_handler.formatter = handler_log_formatter

logger = Logger()


def init_logger():
    logbook.set_datetime_format("local")
    logger.handlers = []
    logger.handlers.append(std_handler)
    # logger.handlers.append(file_handler)


init_logger()
