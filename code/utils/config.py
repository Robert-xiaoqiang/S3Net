import os
import uuid
from datetime import datetime

__all__ = ['proj_root', 'arg_config', 'path_config']

from network.SCFNet import SCFNet_Res50

proj_root = os.path.dirname(os.path.dirname(__file__))
datasets_root = '/home/xqwang/projects/saliency/semi-sod/datasets/'

unlabeled_path = os.path.join(datasets_root, 'SUN-RGBD', 'train_data')
njud_path = os.path.join(datasets_root, 'NJUD', 'test_data')
nlpr_path = os.path.join(datasets_root, 'NLPR', 'test_data')
sip_path = os.path.join(datasets_root, 'SIP')
rgbd135_path = os.path.join(datasets_root, 'RGBD135')
stereo_path = os.path.join(datasets_root, 'STEREO')
lfsd_path = os.path.join(datasets_root, 'LFSD')

# train_path = os.path.join(datasets_root, 'NJUD-NLPR-RGBD135', 'train_data')
# test_path = os.path.join(datasets_root, 'NJUD-NLPR-RGBD135', 'test_data')
train_path = os.path.join(datasets_root, 'NJUD-NLPR', 'train_data')
test_path = os.path.join(datasets_root, 'NJUD-NLPR', 'test_data')

# 配置区域 #####################################################################
arg_config = {
    # 常用配置
    'NET': 'SCFNet_Res50',  # 决定使用哪一个网络
    'SCFNet_Res50': {
        'net': SCFNet_Res50,
        'exp_name': 'SCFNet_Res50'
    },
    
    'only_test': False,
    'resume': True,  # resume when training/testing
    'use_aux_loss': True,  # 是否使用辅助损失
    'save_pre': True,  # 是否保留最终的预测结果
    'epoch_num': 500,  # 训练周期, 0: directly test model
    'lr': 0.001,  # 微调时缩小100倍
    'xlsx_name': 'result.xlsx',
    
    'rgb_data': {
        'unlabeled_path': unlabeled_path,
        'tr_data_path': train_path,
        'te_data_path': test_path,
        'te_data_list': {
            'njud': njud_path,
            'nlpr': nlpr_path,
            'sip': sip_path,
            'rgbd135': rgbd135_path,
            'stereo': stereo_path,
            'lfsd': lfsd_path
        },
    },
    'tb_update': 10,  # >0 则使用tensorboard
    'print_freq': 10,  # >0, 保存迭代过程中的信息
    'prefix': ('.jpg', '.png'),
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名，
    
    'reduction': 'mean',  # 损失处理的方式，可选“mean”和“sum”
    'optim': 'sgd_trick',  # sgd_trick, sgd_r3, sgd_all
    'weight_decay': 5e-4,  # 微调时设置为0.0001
    'momentum': 0.9,
    'nesterov': False,
    'sche_usebatch': False,
    'lr_type': 'poly',
    'lr_decay': 0.9,  # poly
    'use_bigt': True,  # 有时似乎打开好，有时似乎关闭好？
    'batch_size': 8,  # 要是继续训练, 最好使用相同的batchsize
    'num_workers': 8,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    'input_size': 256,
    'gpus': [0, 1],

    'is_mt': None, # set in main.py or main_mt.py
    'labeled_batch_size': 4,
    'ema_decay': 0.99,
    'consistency': 1.0,
    'consistency_rampup': 150.0
}
################################################################################

# summary_key = 'exp-full-channel-so-0' #in output-backup directory: 2 times middle channel njud-nlpr 180 eras / batch 4
# summary_key = 'exp-reduce-channel-mt-0' #: 1 time middle channel njud-nlpr 500 eras / batch 4+4 ? how about 4+2 / 6+2
# summary_key = 'exp-reduce-channel-so-0' #: 1 time middle channel njud-nlpr 180 eras / batch 4
summary_key = 'exp-reduce-channel-so-1' #: 1 time middle channel njud-nlpr 500 eras / batch 8
# summary key solves other varients(supervised only or MT guiding unlabel data)
ckpt_path = os.path.join(os.path.dirname(proj_root), 'output', summary_key)

# this only solves the problem of architecture varients
pth_log_path = os.path.join(ckpt_path, arg_config[arg_config['NET']]['exp_name'])
tb_path = os.path.join(pth_log_path, 'tb')
save_path = os.path.join(pth_log_path, 'pre')
pth_path = os.path.join(pth_log_path, 'pth')

final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth.tar")
final_state_path = os.path.join(pth_path, "state_final.pth")

tr_log_path = os.path.join(pth_log_path, f'tr_{str(datetime.now())[:10]}.txt')
te_log_path = os.path.join(pth_log_path, f'te_{str(datetime.now())[:10]}.txt')
cfg_log_path = os.path.join(pth_log_path, f'cfg_{str(datetime.now())[:10]}.txt')
trainer_log_path = os.path.join(pth_log_path, f'trainer_{str(datetime.now())[:10]}.txt')
xlsx_path = os.path.join(ckpt_path, arg_config['xlsx_name'])

path_config = {
    'ckpt_path': ckpt_path,
    'pth_log': pth_log_path,
    'tb': tb_path,
    'save': save_path,
    'pth': pth_path,
    'final_full_net': final_full_model_path,
    'final_state_net': final_state_path,
    'tr_log': tr_log_path,
    'te_log': te_log_path,
    'cfg_log': cfg_log_path,
    'trainer_log': trainer_log_path,
    'xlsx': xlsx_path
}
