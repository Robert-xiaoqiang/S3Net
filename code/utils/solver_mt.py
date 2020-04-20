import os
import os.path as osp
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam, SGD, lr_scheduler
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from loss.CEL import CEL
from utils.imgs.create_loader_imgs import create_loader
from utils.metric import cal_maxf, cal_pr_mae_meanf, cal_maxe, cal_s
from utils.misc import Ramper, AvgMeter, construct_print, make_log, write_xlsx

class Solver():
    def __init__(self, args, path):
        super(Solver, self).__init__()
        self.args = args
        self.path = path
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.ToPILImage()
        self.model_name = args[args['NET']]['exp_name']
        
        self.tr_data_path = self.args['rgb_data']['tr_data_path']
        self.val_data_path = self.args['rgb_data']['val_data_path']
        self.te_data_path = self.args['rgb_data']['te_data_path']
        self.te_data_list = self.args['rgb_data']['te_data_list']
        self.best_results = {
            'MAE': 100.0,
            'MAXF': 0.0,
            'MAXE': 0.0,
            'S': 0.0,
        }
        
        self.save_path = self.path["save"]
        self.save_pre = self.args["save_pre"]
        if self.args["tb_update"] > 0:
            self.tb = SummaryWriter(self.path["tb"])
        
        # 依赖与前面属性的属性
        self.pth_path = self.path["final_state_net"]
        self.tr_loader = create_loader(
            data_path=self.tr_data_path, mode='train', get_length=False
        )
        self.te_loader, self.te_length = create_loader(
            data_path=self.te_data_path, mode='test', get_length=True
        )
        
        self.net = self.args[self.args["NET"]]["net"]()
        self.net = torch.nn.DataParallel(self.net, device_ids = self.args['gpus'])
        self.net.to(self.dev)
        self.ema_net = self.args[self.args["NET"]]["net"]()
        self.ema_net = torch.nn.DataParallel(self.ema_net, device_ids = self.args['gpus'])
        self.ema_net.to(self.dev)

        for param in self.ema_net.parameters():
            param.detach_()
        self.opti = self.make_optim()
        
        self.end_epoch = self.args["epoch_num"]
        self.iter_num = self.end_epoch * len(self.tr_loader)
        self.sche = self.make_scheduler()
        pprint(self.args)
        
        if self.args['resume']:
            self.resume_checkpoint(load_path=self.path['final_full_net'], mode='all')
        else:
            self.start_epoch = 0
        self.only_test = self.args['only_test']

        if not self.only_test:
            # 损失函数
            self.loss_funcs = [BCELoss(reduction=self.args['reduction']).to(self.dev)]
            if self.args['use_aux_loss']:
                self.loss_funcs.append(CEL(reduction=self.args['reduction']).to(self.dev))
            self.mes_loss = MSELoss(reduction=self.args['reduction']).to(self.dev)
    
    def total_loss(self, train_preds, train_masks):
        loss_list = []
        loss_item_list = []
        
        assert len(self.loss_funcs) != 0, "请指定损失函数`self.loss_funcs`"
        for loss in self.loss_funcs:
            loss_out = loss(train_preds, train_masks)
            loss_list.append(loss_out)
            loss_item_list.append(f"{loss_out.item():.5f}")
        
        train_loss = sum(loss_list)
        return train_loss, loss_item_list
    
    def deep_loss(self, train_preds, train_other_data):
        loss_list = []
        loss_item_list = []
        
        assert len(self.loss_funcs) != 0, "请指定损失函数`self.loss_funcs`"
        for loss in self.loss_funcs:
            loss_out_final = loss(train_preds[0], train_other_data[0])
            # loss_out0 = loss(train_preds[1], train_other_data[1])
            loss_out1 = loss(train_preds[2], train_other_data[2])
            loss_out2 = loss(train_preds[3], train_other_data[3])
            loss_out3 = loss(train_preds[4], train_other_data[4])
            # loss_out4 = loss(train_preds[5], train_other_data[5])
            loss_out = loss_out_final + 0.5 * loss_out1 + 0.3 * loss_out2 + 0.2 * loss_out3
            loss_list.append(loss_out)
            loss_item_list.append(f"{loss_out.item():.5f}")
        
        train_loss = sum(loss_list)
        return train_loss, loss_item_list
    
    def mt_loss(self, train_preds, ema_preds, train_masks, epoch):
        loss_list = []
        loss_item_list = []
        lb = self.args['labeled_batch_size']
        assert len(self.loss_funcs) != 0, "请指定损失函数`self.loss_funcs`"
        for loss in self.loss_funcs:
            loss_out = loss(train_preds[:lb], train_masks[:lb])
            loss_list.append(loss_out)
            loss_item_list.append(f"{loss_out.item():.5f}")
        supervised_loss = sum(loss_list)
        consistency_loss = self.mes_loss(train_preds[lb:], ema_preds)
        consistency_weight = self.get_current_consistency_weight(epoch)
        train_loss = 0.5 * supervised_loss + consistency_weight * consistency_loss
        return train_loss, loss_item_list, supervised_loss, consistency_loss, consistency_weight
    
    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args['consistency'] * Ramper.sigmoid_rampup(epoch, self.args['consistency_rampup'])
    
    def update_ema_variables(self, global_step):
        # Use the true average until the exponential average is more correct
        alpha = self.args['ema_decay']
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_net.parameters(), self.net.parameters()):
            # update params in current device (cpu/cuda:)
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def train(self):
        if not self.only_test:
            self.net.train()
            self.ema_net.train()
            for curr_epoch in range(self.start_epoch, self.end_epoch):
                train_loss_record = AvgMeter()
                for train_batch_id, train_data in enumerate(self.tr_loader):
                    curr_iter = curr_epoch * len(self.tr_loader) + train_batch_id
                    
                    self.opti.zero_grad()
                    train_inputs, train_depth, *train_other_data = train_data
                    train_inputs = train_inputs.to(self.dev, non_blocking=True)
                    train_depth = train_depth.to(self.dev, non_blocking=True)
                    train_other_data = [ d.to(self.dev, non_blocking=True) if torch.is_tensor(d) else d for d in train_other_data ]
                    train_preds = self.net(train_inputs, train_depth)
                    lb = self.args['labeled_batch_size']

                    with torch.no_grad():
                        ema_preds = self.ema_net(train_inputs[lb:], train_depth[lb:])

                    # train_loss, loss_item_list = self.total_loss(train_preds, train_other_data[0])
                    # train_loss, loss_item_list = self.deep_loss(train_preds, train_other_data)
                    train_loss, loss_item_list, \
                    supervised_loss, consistency_loss, consistency_weight \
                    = self.mt_loss(train_preds, ema_preds, train_other_data[0], curr_epoch)
                    train_loss.backward()
                    self.opti.step()
                    self.update_ema_variables(curr_iter)
                    
                    if self.args["sche_usebatch"]:
                        if self.args["lr_type"] == "poly":
                            self.sche.step(curr_iter + 1)
                        else:
                            raise NotImplementedError
                    
                    # 仅在累计的时候使用item()获取数据
                    train_iter_loss = train_loss.item()
                    train_batch_size = train_inputs.size(0)
                    train_loss_record.update(train_iter_loss, train_batch_size)
                    
                    # 显示tensorboard
                    if (self.args["tb_update"] > 0 and (curr_iter + 1) % self.args["tb_update"] == 0):
                        self.tb.add_scalar("data/trloss_avg", train_loss_record.avg, curr_iter)
                        self.tb.add_scalar("data/trloss_iter", train_iter_loss, curr_iter)
                        self.tb.add_scalar("data/trloss_sup", supervised_loss, curr_iter)
                        self.tb.add_scalar("data/trloss_con", consistency_loss, curr_iter)
                        self.tb.add_scalar("data/trlr", self.opti.param_groups[0]["lr"], curr_iter)
                        self.tb.add_scalar("data/trloss_con_weight", consistency_weight, curr_iter)
                        
                        lb = self.args["labeled_batch_size"]
                        tr_tb_mask = make_grid(train_other_data[0][:lb], nrow=lb, padding=5)
                        self.tb.add_image("trmasks_labeled", tr_tb_mask, curr_iter)

                        tr_tb_out_1 = make_grid(train_preds[:lb], nrow=lb, padding=5)
                        self.tb.add_image("trpreds_labeled", tr_tb_out_1, curr_iter)
                        tr_tb_out_2 = make_grid(train_preds[lb:], nrow=train_batch_size - lb, padding=5)
                        self.tb.add_image("trpreds_unlabeled", tr_tb_out_2, curr_iter)
                        tr_tb_out_3 = make_grid(ema_preds, nrow=train_batch_size - lb, padding=5)
                        self.tb.add_image("emapreds_unlabeled", tr_tb_out_3, curr_iter)
                    # 记录每一次迭代的数据
                    if (self.args["print_freq"] > 0 and (curr_iter + 1) % self.args["print_freq"] == 0):
                        log = (
                            f"[I:{curr_iter}/{self.iter_num}][E:{curr_epoch}:{self.end_epoch}]>"
                            f"[{self.model_name}]"
                            f"[Lr:{self.opti.param_groups[0]['lr']:.7f}]"
                            f"[Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}|"
                            f"0.5*{loss_item_list}+({consistency_weight:.5f}*{consistency_loss:.5f})]"
                        )
                        print(log)
                        make_log(self.path["tr_log"], log)
                
                # 根据周期修改学习率
                if not self.args["sche_usebatch"]:
                    if self.args["lr_type"] == "poly":
                        self.sche.step(curr_epoch + 1)
                    else:
                        raise NotImplementedError
                
                # 每个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
                self.save_checkpoint(
                    curr_epoch + 1,
                    full_net_path=self.path['final_full_net'],
                    state_net_path=self.path['final_state_net']
                )  # 保存参数
                self.validate(curr_epoch)

        total_results = {}
        for data_name, data_path in self.te_data_list.items():
            construct_print(f"Testing with testset: {data_name}")
            self.te_loader, self.te_length = create_loader(
                data_path=data_path, mode='test', get_length=True
            )
            self.save_path = os.path.join(self.path["save"], data_name)
            if not os.path.exists(self.save_path):
                construct_print(f"{self.save_path} do not exist. Let's create it.")
                os.makedirs(self.save_path)
            results = self.test(save_pre=self.save_pre)
            msg = (f"Results on the testset({data_name}:'{data_path}'): {results}")
            construct_print(msg)
            make_log(self.path["te_log"], msg)
            
            total_results[data_name.upper()] = results
        # save result into xlsx file.
        # write_xlsx(self.model_name, total_results)

    def validate(self, curr_epoch):
        self.te_loader, self.te_length = create_loader(
            data_path=self.val_data_path, mode='test', get_length=True
        )
        construct_print('Start validating')
        results = self.test(save_pre=False) # just flush it
        if results['S'] > self.best_results['S'] or \
           results['MAXF'] > self.best_results['MAXF'] or \
           results['MAXE'] > self.best_results['MAXE'] or \
           results['MAE'] < self.best_results['MAE']:
           self.best_results.update(results)
           self.save_checkpoint(
                        curr_epoch + 1,
                        full_net_path=self.path['final_full_net'],
                        state_net_path=self.path['final_state_net'],
                        save_key = 'best'
                    )
           construct_print('Update best epoch')
           construct_print('Epoch {} with best validating results: {}'.format(curr_epoch, self.best_results))
        construct_print('Finish validating')

    def test(self, save_pre):
        if self.only_test:
            self.resume_checkpoint(load_path=self.pth_path, mode='onlynet')
        self.net.eval()
        
        loader = self.te_loader
        
        pres = [AvgMeter() for _ in range(256)]
        recs = [AvgMeter() for _ in range(256)]
        meanfs = AvgMeter()
        maes = AvgMeter()
        maxes = AvgMeter()
        ss = AvgMeter()
        
        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave=False)
        for test_batch_id, test_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.model_name}: te=>{test_batch_id + 1}")
            with torch.no_grad():
                in_imgs, in_depths, in_mask_paths, in_names = test_data
                in_imgs = in_imgs.to(self.dev, non_blocking=True)
                in_depths = in_depths.to(self.dev, non_blocking=True)
                outputs = self.net(in_imgs, in_depths)
            
            outputs_np = outputs.cpu().detach()
            
            for item_id, out_item in enumerate(outputs_np):
                gimg_path = osp.join(in_mask_paths[item_id])
                ###########################################
                gt_img = Image.open(gimg_path).convert("L") # be careful
                ###########################################
                out_img = self.to_pil(out_item).resize(gt_img.size)
                
                if save_pre:
                    oimg_path = osp.join(self.save_path, in_names[item_id] + ".png")
                    out_img.save(oimg_path)
                
                gt_img = np.asarray(gt_img)
                out_img = np.array(out_img)
                ps, rs, mae, meanf = cal_pr_mae_meanf(out_img, gt_img)
                for pidx, pdata in enumerate(zip(ps, rs)):
                    p, r = pdata
                    pres[pidx].update(p)
                    recs[pidx].update(r)
                maes.update(mae)
                meanfs.update(meanf)
                
                maxe = cal_maxe(out_img, gt_img)
                maxes.update(maxe)

                s = cal_s(out_img, gt_img)
                ss.update(s)


        maxf = cal_maxf([pre.avg for pre in pres], [rec.avg for rec in recs])
        results = {"MAXF": maxf, "MEANF": meanfs.avg, 
                   "MAXE": maxes.avg, "S": ss.avg,
                   "MAE": maes.avg}
        return results
    
    def make_scheduler(self):
        total_num = self.iter_num if self.args['sche_usebatch'] else self.end_epoch
        if self.args["lr_type"] == "poly":
            lamb = lambda curr: pow((1 - float(curr) / total_num), self.args["lr_decay"])
            scheduler = lr_scheduler.LambdaLR(self.opti, lr_lambda=lamb)
        else:
            raise NotImplementedError
        return scheduler
    
    def make_optim(self):
        if self.args["optim"] == "sgd_trick":
            # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
            params = [
                {
                    "params": [
                        p for name, p in self.net.named_parameters()
                        if ("bias" in name or "bn" in name)
                    ],
                    "weight_decay":
                        0,
                },
                {
                    "params": [
                        p for name, p in self.net.named_parameters()
                        if ("bias" not in name and "bn" not in name)
                    ]
                },
            ]
            optimizer = SGD(
                params,
                lr=self.args["lr"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
                nesterov=self.args["nesterov"]
            )
        elif self.args["optim"] == "sgd_r3":
            params = [
                # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
                # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
                # 到减少模型过拟合的效果。
                {
                    "params":
                        [param for name, param in self.net.named_parameters() if
                         name[-4:] == "bias"],
                    "lr":
                        2 * self.args["lr"],
                },
                {
                    "params":
                        [param for name, param in self.net.named_parameters() if
                         name[-4:] != "bias"],
                    "lr":
                        self.args["lr"],
                    "weight_decay":
                        self.args["weight_decay"],
                },
            ]
            optimizer = SGD(params, momentum=self.args["momentum"])
        elif self.args["optim"] == "sgd_all":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"],
                momentum=self.args["momentum"]
            )
        elif self.args["optim"] == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.args["lr"],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args["weight_decay"]
            )
        else:
            raise NotImplementedError
        print("optimizer = ", optimizer)
        return optimizer
    
    def save_checkpoint(self, current_epoch, full_net_path, state_net_path, save_key = None):
        """
        保存完整参数模型（大）和状态参数模型（小）

        Args:
            current_epoch (int): 当前周期
            full_net_path (str): 保存完整参数模型的路径
            state_net_path (str): 保存模型权重参数的路径
        """
        state_dict = {
            'arch': self.args["NET"],
            'epoch': current_epoch,
            'net_state': self.net.state_dict(),
            'ema_net_state': self.ema_net.state_dict(),
            'opti_state': self.opti.state_dict(),
            'sche_state': self.sche.state_dict()
        }
        torch.save(state_dict, full_net_path)
        torch.save(self.net.state_dict(), state_net_path)

        if save_key is not None:
            dirname, basename0 = os.path.split(full_net_path)
            dirname, basename1 = os.path.split(state_net_path)
            new_dirname = os.path.join(dirname, save_key)
            os.makedirs(new_dirname, exist_ok = True)
            new_filename0 = os.path.join(new_dirname, basename0)
            new_filename1 = os.path.join(new_dirname, basename1)
            torch.save(state_dict, new_filename0)
            torch.save(self.net.state_dict(), new_filename1)            
    
    def resume_checkpoint(self, load_path, mode='all'):
        """
        从保存节点恢复模型

        Args:
            load_path (str): 模型存放路径
            mode (str): 选择哪种模型恢复模式:
                - 'all': 回复完整模型，包括训练中的的参数；
                - 'onlynet': 仅恢复模型权重参数
        """
        if os.path.exists(load_path) and os.path.isfile(load_path):
            construct_print(f"Loading checkpoint '{load_path}'")
            checkpoint = torch.load(load_path)
            if mode == 'all':
                if self.args["NET"] == checkpoint['arch']:
                    self.start_epoch = checkpoint['epoch']
                    self.net.load_state_dict(checkpoint['net_state'])
                    self.ema_net.load_state_dict(checkpoint['ema_net_state'])
                    self.opti.load_state_dict(checkpoint['opti_state'])
                    self.sche.load_state_dict(checkpoint['sche_state'])
                    construct_print(f"Loaded '{load_path}' "
                                    f"(epoch {checkpoint['epoch']})")
                else:
                    raise Exception(f"{load_path} does not match.")
            elif mode == 'onlynet':
                self.net.load_state_dict(checkpoint)
                construct_print(f"Loaded checkpoint '{load_path}' "
                                f"(only has the net's weight params)")
            else:
                raise NotImplementedError
        else:
            self.start_epoch = 0
            construct_print(f'Cannot found pth in {load_path:}, then train from(test based on) scratch')
