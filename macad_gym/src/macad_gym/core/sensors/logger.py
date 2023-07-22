import sys, os
import logging
from macad_gym import LOG_PATH

class Logger:
    def __init__(self, name, path = None, Flevel = None, Clevel = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.Flevel = Flevel
        self.Clevel = Clevel
        self.fmt = logging.Formatter('[%(levelname)s] %(name)s [%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        #self.fmt = logging.Formatter('[%(levelname)s] %(name)s [%(process)d %(thread)d] [%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set command line logging
        if Clevel is not None:
            self.sh = logging.StreamHandler(sys.stdout)
            self.sh.setFormatter(self.fmt)
            self.sh.setLevel(Clevel)
            self.logger.addHandler(self.sh)
        # set file logging
        if path is not None:
            self.fh = logging.FileHandler(path)
            self.fh.setFormatter(self.fmt)
            self.fh.setLevel(Flevel)
            self.logger.addHandler(self.fh) 
    
    def reset_file(self, path):
        assert path is not None
        self.fh.flush()
        self.sh.flush()
        fh = self.fh
        self.logger.removeHandler(self.fh)
        self.fh = logging.FileHandler(path)
        self.fh.setFormatter(self.fmt)
        self.fh.setLevel(self.Flevel)
        self.logger.addHandler(self.fh)

        return fh
 
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

class LOG:
    log_dir = None
    log_file = None

    reward_logger = None
    multi_env_logger = None
    pdqn_logger = None
    hud_logger = None
    basic_agent_logger = None
    pdqn_multi_agent_logger = None
    route_planner_logger = None
    misc_logger = None
    traffic_logger = None

    @staticmethod 
    def set_log(path, file_name = None):
        LOG.log_dir = path
        if not os.path.exists(LOG.log_dir):
            os.makedirs(LOG.log_dir)
        if file_name is None:
            LOG.log_file = LOG.log_dir + '/macad-gym.log'
        else:
            LOG.log_file = LOG.log_dir + file_name

        # set each logger
        if LOG.traffic_logger is None:
            LOG.traffic_logger = Logger('traffic.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.traffic_logger.reset_file(LOG.log_file)
        if LOG.misc_logger is None:
            LOG.misc_logger = Logger('misc.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.misc_logger.reset_file(LOG.log_file)
        if LOG.route_planner_logger is None:
            LOG.route_planner_logger = Logger('route_planner.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.route_planner_logger.reset_file(LOG.log_file)
        if LOG.reward_logger is None:
            LOG.reward_logger = Logger('reward.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.reward_logger.reset_file(LOG.log_file)
        if LOG.multi_env_logger is None:
            LOG.multi_env_logger = Logger('multi_env.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.multi_env_logger.reset_file(LOG.log_file)
        if LOG.pdqn_logger is None:
            LOG.pdqn_logger = Logger('pdqn.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.pdqn_logger.reset_file(LOG.log_file)
        if LOG.hud_logger is None:
            LOG.hud_logger = Logger('hud.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.hud_logger.reset_file(LOG.log_file)
        if LOG.basic_agent_logger is None:
            LOG.basic_agent_logger = Logger('basic_agent.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            LOG.basic_agent_logger.reset_file(LOG.log_file)
        if LOG.pdqn_multi_agent_logger is None:
            LOG.pdqn_multi_agent_logger = Logger('pdqn_multi_agent.py', LOG.log_file, logging.DEBUG, logging.ERROR)
        else:
            fh = LOG.pdqn_multi_agent_logger.reset_file(LOG.log_file)
            fh.close()
            

if LOG.log_dir is None:
    LOG.set_log(LOG_PATH + '/0')
