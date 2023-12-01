import shutil
import inspect
import numpy as np
import torch
import random
import os
import logging
import sys
from Retrain import args


def get_init_info(model, dataset, model_name, LOGGER):
    init_params = inspect.signature(model.__init__).parameters

    LOGGER.info(f'datset:{args.dataset}; \n'
                f'num:{len(dataset)}; \n'
                f'model:{model_name}; \n'
                f'batchsize:{args.batch_size}; \n')
    for param in init_params.values():
        para_value = getattr(model, param.name)
        LOGGER.info(f'{param}:{para_value}\n')


def get_super_info(model, dataset, model_name, LOGGER):
    init_params = inspect.signature(model.__init__).parameters

    LOGGER.info(f'datset:{args.dataset}; \n'
                f'num:{len(dataset)}; \n'
                f'model:{model_name}; \n'
                f'batchsize:{args.batch_size}; \n')
    for param in init_params.values():
        para_value = getattr(model, param.name)
        LOGGER.info(f'{param}:{para_value}\n')


def get_evo_info(model, dataset, model_name, LOGGER):
    init_params = inspect.signature(model.__init__).parameters

    LOGGER.info(f'datset:{args.dataset}; \n'
                f'num:{len(dataset)}; \n'
                f'model:{model_name}; \n'
                f'batchsize:{args.batch_size}; \n')
    for param in init_params.values():
        para_value = getattr(model, param.name)
        LOGGER.info(f'{param}:{para_value}\n')


def get_retrain_info(model, dataset, model_name, LOGGER):
    init_params = inspect.signature(model.__init__).parameters

    LOGGER.info(f'datset:{args.dataset}; \n'
                f'num:{len(dataset)}; \n'
                f'model:{model_name}; \n'
                f'batchsize:{args.batch_size}; \n')
    for param in init_params.values():
        para_value = getattr(model, param.name)
        LOGGER.info(f'{param}:{para_value}\n')


def logger_path(model_name, dataset, stacked_num, concat_mlp, run_time, ROOT_PATH):
    log_root_path = os.path.join(ROOT_PATH, 'log')
    if stacked_num >= 0:
        stack = 'stacked'
        stacked_num = str(stacked_num)
        if concat_mlp is True:
            concat = 'concat'
        else:
            concat = 'no concat'
        log_parent_dir = os.path.join(log_root_path,
                                      dataset,
                                      str(args.seed),
                                      model_name,
                                      stack,
                                      stacked_num,
                                      concat,
                                      run_time,
                                      )
    else:
        stack = 'original'
        log_parent_dir = os.path.join(log_root_path,
                                      dataset,
                                      str(args.seed),
                                      model_name,
                                      stack,
                                      run_time,
                                      )
    check_directory(log_parent_dir, force_removed=True)
    log_save_dir = os.path.join(log_parent_dir,
                                f'{model_name}_{run_time}.txt')
    return log_save_dir


def get_max(auc_list):
    value = max(auc_list)
    idx = auc_list.index(value)
    return value, idx


def get_min(loss_list):
    value = min(loss_list)
    idx = loss_list.index(value)
    return value, idx

def get_average(auc_list):
    sum = 0
    for item in auc_list:
        sum += item
    return sum / len(auc_list)


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters_in_mb(model):
    res = 0

    for name, p in model.named_parameters():
        if "auxiliary" not in name and p is not None and p.requires_grad:
            res += p.numel()
    return res / 1e6


class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.loss = 100
        self.save_path = save_path

    def is_continue(self, model, accuracy, loss):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            if loss < self.loss:
                self.loss = loss
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def logger_result(times, auc_list, loss_list, model_name, LOGGER):
    value_auc, idx_auc = get_max(auc_list)
    average_auc = get_average(auc_list)
    value_loss, idx_loss = get_min(loss_list)
    average_loss = get_average(loss_list)
    LOGGER.info(f'\n{model_name}\n'
                f'total {times} experiments\n'
                f'the output is below：\n'
                f'auc:{auc_list}\n'
                f'loss:{loss_list}\n'
                f'The biggest value of auc is : {value_auc}，in the number of {idx_auc} experiment\n'
                f'The smallest loss is : {value_loss}，in the number of {idx_loss} experiment\n'
                f'The average of auc is : {average_auc}\n'
                f'The average of loss is : {average_loss}\n')


def check_directory(path, force_removed=False):
    if force_removed:
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass

    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(name, save_dir=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if save_dir is None:
        return logger
    log_fmt = '%(asctime)s.%(msecs)03d %(levelname)s : %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, level=level, format=log_fmt, datefmt=date_fmt)
    temp = os.path.split(save_dir)
    debug_save_dir = os.path.join(temp[0], temp[1])
    fh_debug = logging.FileHandler(debug_save_dir)
    debug_filter = logging.Filter()
    debug_filter.filter = lambda record: record.levelno >= level
    fh_debug.addFilter(debug_filter)
    fh_debug.setFormatter(logging.Formatter(log_fmt, date_fmt))
    logger.addHandler(fh_debug)
    return logger


def set_logger(model_name, dataset, stacked_num, concat_mlp, run_time, ROOT_PATH):
    log_save_dir = logger_path(model_name, dataset, stacked_num, concat_mlp, run_time, ROOT_PATH)
    LOGGER = get_logger(run_time, log_save_dir)
    LOGGER.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    return LOGGER


def get_log(name=""):
    FORMATTER = logging.Formatter(fmt="[{asctime}]:{message}", style= '{')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)
    return logger