import sys, os
import logging
from macad_gym import LOG_PATH, RETRIES_ON_ERROR


class Logger(object):
    def __init__(self, name, path = None, Flevel = None, Clevel = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.Flevel = Flevel
        self.Clevel = Clevel
        self.path = path
        
        self.add_handlers(path)
        # weak_self = weakref.ref(self)
        # for i in range(RETRIES_ON_ERROR):
        #     out = Logger.add_handlers(weak_self)
        #     if out:
        #         break
        #     else:
        #         raise UserWarning(f"Logger Add Handlers Failed, Path:{path}, Retry Times:{i}")

    def reset_file(self, path):
        assert path is not None
        while self.logger.hasHandlers():
            if isinstance(self.logger.handlers[0], logging.FileHandler):
                self.logger.handlers[0].close()
            self.logger.removeHandler(self.logger.handlers[0])

        self.add_handlers(path)
 
    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)
 
    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
 
    def warning(self, message, *args, **kwargs):
        self.logger.warn(message, *args, **kwargs)

    def warn(self, message, *args, **kwargs):
        self.logger.warn(message, *args, **kwargs)
  
    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
 
    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self.logger.exception(message, *args, **kwargs)

    def add_handlers(self, path):        
        fmt = logging.Formatter('[%(levelname)s] %(name)s [%(process)d %(thread)d] [%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        #self.fmt = logging.Formatter('[%(levelname)s] %(name)s [%(process)d %(thread)d] [%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set command line logging
        if self.Clevel is not None:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            sh.setLevel(self.Clevel)
            self.logger.addHandler(sh)
        # set file logging
        if path is not None:
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            fh.setLevel(self.Flevel)
            self.logger.addHandler(fh) 

        return True

class LOG(object):
    log_dir = None
    log_file = None
    server_log = None

    reward_logger = None
    multi_env_logger = None
    pdqn_logger = None
    psac_logger = None
    hud_logger = None
    basic_agent_logger = None
    rl_trainer_logger = None
    route_planner_logger = None
    misc_logger = None
    traffic_logger = None
    camera_manager_logger = None
    derived_sensors_logger = None

    @staticmethod 
    def set_log(path, file_name:str = None):
        LOG.log_dir = path
        if not os.path.exists(LOG.log_dir):
            os.makedirs(LOG.log_dir)
        if file_name is None:
            LOG.log_file = os.path.join(LOG.log_dir, 'macad-gym.log')
        else:
            LOG.log_file = os.path.join(LOG.log_dir, file_name)

        if LOG.derived_sensors_logger is None:
            LOG.derived_sensors_logger = Logger('derived_sensors.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.camera_manager_logger = Logger('camera_manager.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.traffic_logger = Logger('traffic.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.misc_logger = Logger('misc.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.route_planner_logger = Logger('route_planner.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.reward_logger = Logger('reward.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.multi_env_logger = Logger('multi_env.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.pdqn_logger = Logger('pdqn.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.psac_logger = Logger('psac.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.hud_logger = Logger('hud.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.basic_agent_logger = Logger('basic_agent.py', LOG.log_file, logging.DEBUG, logging.ERROR)
            LOG.rl_trainer_logger = Logger('rl_trainer.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            attrs = vars(LOG)
            for attr, value in attrs.items():
                if isinstance(value, Logger):
                    [handler.flush() for handler in value.logger.handlers]

            for attr, value in attrs.items():
                if isinstance(value, Logger):
                    value.reset_file(LOG.log_file)


if LOG.log_dir is None:
    LOG.server_log = os.path.join(LOG_PATH, 'carla_server.log')
    LOG.set_log(os.path.join(LOG_PATH, '0'))