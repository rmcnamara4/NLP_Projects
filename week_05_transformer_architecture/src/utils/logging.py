import logging 
import os

def setup_logging(log_dir = './logs', log_file = 'train.log', filemode = 'w'): 
    os.makedirs(log_dir, exist_ok = True) 
    log_path = os.path.join(log_dir, log_file) 

    file_handler = logging.FileHandler(log_path, mode = filemode)

    logging.basicConfig(
        level = logging.INFO, 
        format = '%(asctime)s - [%(levelname)s] %(message)s', 
        handlers = [
            file_handler, 
            logging.StreamHandler()
        ]
    )