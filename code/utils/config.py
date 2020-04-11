import os
from datetime import datetime

__all__ = ['proj_root', 'arg_config', 'path_config']

from network.MINet import MINet_RGBD_Res50
# code root directory
proj_root = os.path.dirname(os.path.dirname(__file__))
datasets_root = '/home/xqwang/projects/saliency/semi-sod/datasets/'

njud_path = os.path.join(datasets_root, 'NJUD', 'test_data')
nlpr_path = os.path.join(datasets_root, 'NLPR', 'test_data')
sip_path = os.path.join(datasets_root, 'SIP')
rgbd135_path = os.path.join(datasets_root, 'RGBD135')
stereo_path = os.path.join(datasets_root, 'STEREO')
lfsd_path = os.path.join(datasets_root, 'LFSD')

train_path = os.path.join(datasets_root, 'NJUD-NLPR', 'train_data')
test_path = os.path.join(datasets_root, 'NJUD-NLPR', 'test_data')

# 配置区域 #####################################################################
arg_config = {
    # 常用配置
    'NET': 'MINet_RGBD_Res50',  # 决定使用哪一个网络
    'MINet_RGBD_Res50': {
        'net': MINet_RGBD_Res50,
        'exp_name': 'MINet_RGBD_Res50'
    },
     
    'resume': False,  # 是否需要恢复模型
    'use_aux_loss': True,  # 是否使用辅助损失
    'save_pre': True,  # 是否保留最终的预测结果
    'epoch_num': 20,  # 训练周期, 0: directly test model
    'lr': 0.001,  # 微调时缩小100倍
    'xlsx_name': 'result.xlsx',
    
    'rgb_data': {
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
    'tb_update': 0,  # >0 则使用tensorboard
    'print_freq': 10,  # >0, 保存迭代过程中的信息
    'prefix': ('.jpg', '.png'),
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名，
    
    'reduction': 'mean',  # 损失处理的方式，可选“mean”和“sum”
    'optim': 'sgd_trick',  # 自定义部分的学习率
    'weight_decay': 5e-4,  # 微调时设置为0.0001
    'momentum': 0.9,
    'nesterov': False,
    'sche_usebatch': False,
    'lr_type': 'poly',
    'lr_decay': 0.9,  # poly
    'use_bigt': True,  # 有时似乎打开好，有时似乎关闭好？
    'batch_size': 4,  # 要是继续训练, 最好使用相同的batchsize
    'num_workers': 8,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    'input_size': 256,
}
################################################################################

ckpt_path = os.path.join(os.path.dirname(proj_root), 'output')

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
