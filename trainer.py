from __future__ import annotations

from torch import optim
from torch.autograd import Variable
from torch.multiprocessing import Manager
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

from config import cfg
from misc.utils import *
from models.CC import CrowdCounter

from PIL.Image import Image
from typing import Any, List
from datasets import DataDict
from easydict import EasyDict
from misc.transforms import Compose
from torch.utils.data import DataLoader
from multiprocessing.managers import DictProxy
from models.layer import CyclicLRWithRestarts, CosineAnnealingWarmupRestarts
import train
from iafoule.metrics import get_metrics


class Trainer:
    def __init__(self, dataloader: Any, cfg_data: EasyDict, pwd: str | bytes):
        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd: str | bytes = pwd

        self.device = cfg.DEVICE

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID, self.net_name).to(self.device)

        optimizer = cfg.OPTIM.lower()
        stepper = cfg.SCHEDULER.lower()
        if optimizer == "adam":
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
            )
        elif optimizer == "nadam":
            self.optimizer = optim.NAdam(
                self.net.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
            )
        elif optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.net.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
            )
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.net.parameters(),
                cfg.LR,
                momentum=0.95,
                weight_decay=cfg.WEIGHT_DECAY,
            )
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        if stepper == "annealing":
            # self.scheduler = CyclicLRWithRestarts(
            #     self.optimizer, cfg_data.TRAIN_BATCH_SIZE, cfg.MAX_EPOCH, restart_period=50, t_mult=1.3
            # )
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 200, 2)
        elif stepper == "custom_annealing":
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                200,
                2,
                warmup_steps=50,
                gamma=0.5,
                min_lr=cfg.LR / 1e4,
                max_lr=cfg.LR,
            )
        else:
            self.scheduler = StepLR(
                self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY
            )

        self.train_record = {"best_mae": 1e20, "best_mse": 1e20, "best_model_name": ""}
        self.timer = {"iter time": Timer(), "train time": Timer(), "val time": Timer()}
        self.writer, self.log_txt = logger(
            self.exp_path, self.exp_name, self.pwd, "exp"
        )

        self.scaler = torch.amp.GradScaler(self.device.type)

        # Determine dtype for autocast (computed once)
        self.amp_dtype = torch.float16
        if self.device.type == "cuda" and torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8:
            self.amp_dtype = torch.bfloat16

        self.i_tb = 0
        self.epoch = -1

        if cfg.PRE_GCC:
            self.net.load_state_dict(
                torch.load(cfg.PRE_GCC_MODEL, map_location=self.device)
            )

        self.train_loader: DataLoader[Any] | None
        self.val_loader: DataLoader[Any]
        self.restore_transform: Compose
        """
        if self.data_mode in ["SHHA", "SHHB", "QNRF", "UCF50"]:
            if cfg.DATA_WORKERS == 0:
                datas: DictProxy[str, List[Image]] | DataDict = {}
            else:
                self.manager = Manager()
                datas = self.manager.dict()
            self.train_loader, self.val_loader, self.restore_transform = dataloader(
                datas, data_workers=cfg.DATA_WORKERS
            )
        else:
            self.train_loader, self.val_loader, self.restore_transform = dataloader()
        """
        multiple_loading = self.data_mode in ["SHHA", "SHHB", "QNRF", "UCF50"]
        if hasattr(cfg_data, "MULTIPLE_DATALOADER"):
            multiple_loading = cfg_data.MULTIPLE_DATALOADER
        if multiple_loading:
            if cfg.DATA_WORKERS == 0:
                train_datas: DictProxy[str, List[Image]] | DataDict = {}
                val_datas: DictProxy[str, List[Image]] | DataDict = {}
            else:
                self.train_manager = Manager()
                train_datas = self.train_manager.dict()
                self.val_manager = Manager()
                val_datas = self.val_manager.dict()
            self.train_loader, self.val_loader, self.restore_transform = dataloader(
                train_datas, val_datas, data_workers=cfg.DATA_WORKERS
            )
        else:
            self.train_loader, self.val_loader, self.restore_transform = dataloader()

    def forward(self):
        # self.validate_V1()
        for epoch in range(cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

            # training
            self.timer["train time"].tic()
            self.train()
            self.timer["train time"].toc(average=False)

            print("train time: {:.2f}s".format(self.timer["train time"].diff))
            print("=" * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer["val time"].tic()
                if self.data_mode == "WE":
                    self.validate_V2()
                elif self.data_mode == "GCC":
                    self.validate_V3()
                else:
                    self.validate_V1()
                self.timer["val time"].toc(average=False)
                print("val time: {:.2f}s".format(self.timer["val time"].diff))

    def train(self):  # training for all datasets
        train_losses = AverageMeter()
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer["iter time"].tic()
            img = Variable(data[0]).to(self.device)
            gt_map = Variable(data[1]).to(self.device)

            sample_weight = None
            if len(data) == 3:
                sample_weight = Variable(data[2]).to(self.device)

            self.optimizer.zero_grad()
            if cfg.USE_AMP_TRAIN:
                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype):
                    pred_map = self.net(img, gt_map, sample_weight)
                    loss = self.net.loss
            else:
                pred_map = self.net(img, gt_map, sample_weight)
                loss = self.net.loss

            if isinstance(self.net.lc_loss, int):
                lc_loss = self.net.lc_loss
            else:
                lc_loss = self.net.lc_loss.item()

            if cfg.USE_AMP_TRAIN:
                # Disable scaler for bfloat16 (CUDA) as it doesn't need scaling
                if self.device.type == "cuda" and self.amp_dtype == torch.bfloat16:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                    self.optimizer.step()
                else:
                    # Use scaler for float16 (CUDA or MPS)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar("train_loss", loss.item(), self.i_tb)
                self.timer["iter time"].toc(average=False)
                print(
                    "[ep %d][it %d][loss %.4f][lc_loss %.4f][lr %.4f][%.2fs]"
                    % (
                        self.epoch + 1,
                        i + 1,
                        loss.item(),
                        lc_loss,
                        self.optimizer.param_groups[0]["lr"] * 10000,
                        self.timer["iter time"].diff,
                    )
                )
                print(
                    "        [cnt: gt: %.1f pred: %.2f]"
                    % (
                        gt_map[0].sum().detach() / self.cfg_data.LOG_PARA,
                        pred_map[0].sum().detach() / self.cfg_data.LOG_PARA,
                    )
                )
                train_losses.update(loss)
            train_loss = train_losses.avg
            self.writer.add_scalar("train_loss", train_loss, self.epoch + 1)

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mapes = AverageMeter()
        mses = AverageMeter()

        time_sampe = 0
        step = 0

        with torch.no_grad():
            for vi, data in enumerate(self.val_loader, 0):
                sample_weight = None
                img = Variable(data[0]).to(self.device)
                gt_map = Variable(data[1]).to(self.device)
                if len(data) == 3:
                    sample_weight = Variable(data[2]).to(self.device)

                if cfg.USE_AMP_VAL:
                    with torch.amp.autocast(self.device.type, dtype=self.amp_dtype):
                        pred_map = self.net.forward(img, gt_map, sample_weight)
                else:
                    pred_map = self.net.forward(img, gt_map, sample_weight)

                step = step + 1
                time_start1 = time.time()
                test_map = self.net.test_forward(img)
                time_end1 = time.time()
                time_sampe += time_end1 - time_start1

                pred_map = pred_map.detach().cpu().numpy()
                gt_map = gt_map.data.detach().cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    losses.update(self.net.loss.item())

                    metrics = get_metrics(
                        pred_map[i_img].squeeze() / self.cfg_data.LOG_PARA,
                        gt_map[i_img] / self.cfg_data.LOG_PARA,
                    )

                    maes.update(metrics["absolute_error"])
                    mapes.update(metrics["absolute_percentage_error"])
                    mses.update(metrics["squared_error"])

                if vi == -1:
                    vis_results(
                        self.exp_name,
                        self.epoch,
                        self.writer,
                        self.restore_transform,
                        img,
                        pred_map,
                        gt_map,
                    )

        mae = maes.avg
        mape = mapes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar("val_loss", loss, self.epoch + 1)
        self.writer.add_scalar("mae", mae, self.epoch + 1)
        self.writer.add_scalar("mape", mape, self.epoch + 1)
        self.writer.add_scalar("rmse", mse, self.epoch + 1)

        self.train_record = update_model(
            self.net,
            self.epoch,
            self.exp_path,
            self.exp_name,
            [mae, mse, loss],
            self.train_record,
            self.log_txt,
        )
        print_summary(self.exp_name, [mae, mse, loss], self.train_record)
        print("\nForward Time: %fms" % (time_sampe * 1000 / step))

    def validate_V2(self):  # validate_V2 for WE
        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        roi_mask = []
        from datasets.WE.setting import cfg_data
        from scipy import io as sio  # type: ignore

        for val_folder in cfg_data.VAL_FOLDER:
            roi_mask.append(
                sio.loadmat(
                    os.path.join(cfg_data.DATA_PATH, "test", val_folder + "_roi.mat")
                )["BW"]
            )

        for i_sub, i_loader in enumerate(self.val_loader, 0):
            # mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).to(self.device)
                    gt_map = Variable(gt_map).to(self.device)

                    if cfg.USE_AMP_VAL:
                        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype):
                            pred_map = self.net.forward(img, gt_map)
                    else:
                        pred_map = self.net.forward(img, gt_map)

                    pred_map = pred_map.detach().cpu().numpy()
                    gt_map = gt_map.detach().cpu().numpy()

                    for i_img in range(pred_map.shape[0]):
                        pred_cnt = pred_map[i_img].sum() / self.cfg_data.LOG_PARA
                        gt_count = gt_map[i_img].sum() / self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(), i_sub)
                        maes.update(abs(gt_count - pred_cnt), i_sub)
                    if vi == 0:
                        vis_results(
                            self.exp_name,
                            self.epoch,
                            self.writer,
                            self.restore_transform,
                            img,
                            pred_map,
                            gt_map,
                        )

        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar("val_loss", loss, self.epoch + 1)
        self.writer.add_scalar("mae", mae, self.epoch + 1)
        self.writer.add_scalar("mae_s1", maes.avg[0], self.epoch + 1)
        self.writer.add_scalar("mae_s2", maes.avg[1], self.epoch + 1)
        self.writer.add_scalar("mae_s3", maes.avg[2], self.epoch + 1)
        self.writer.add_scalar("mae_s4", maes.avg[3], self.epoch + 1)
        self.writer.add_scalar("mae_s5", maes.avg[4], self.epoch + 1)

        self.train_record = update_model(
            self.net,
            self.epoch,
            self.exp_path,
            self.exp_name,
            [mae, 0, loss],
            self.train_record,
            self.log_txt,
        )
        print_WE_summary(
            self.log_txt, self.epoch, [mae, 0, loss], self.train_record, maes
        )

    def validate_V3(self):  # validate_V3 for GCC
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {
            "level": AverageCategoryMeter(9),
            "time": AverageCategoryMeter(8),
            "weather": AverageCategoryMeter(7),
        }
        c_mses = {
            "level": AverageCategoryMeter(9),
            "time": AverageCategoryMeter(8),
            "weather": AverageCategoryMeter(7),
        }

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).to(self.device)
                gt_map = Variable(gt_map).to(self.device)

                if cfg.USE_AMP_VAL:
                    with torch.amp.autocast(self.device.type, dtype=self.amp_dtype):
                        pred_map = self.net.forward(img, gt_map)
                else:
                    pred_map = self.net.forward(img, gt_map)

                pred_map = pred_map.detach().cpu().numpy()
                gt_map = gt_map.detach().cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = pred_map[i_img].sum() / self.cfg_data.LOG_PARA
                    gt_count = gt_map[i_img].sum() / self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)
                    # attributes_pt = attributes_pt.squeeze()
                    # c_maes['level'].update(s_mae, attributes_pt[i_img][0])
                    # c_mses['level'].update(s_mse, attributes_pt[i_img][0])
                    # c_maes['time'].update(s_mae, attributes_pt[i_img][1] / 3)
                    # c_mses['time'].update(s_mse, attributes_pt[i_img][1] / 3)
                    # c_maes['weather'].update(s_mae, attributes_pt[i_img][2])
                    # c_mses['weather'].update(s_mse, attributes_pt[i_img][2])

                # if vi == 0:
                #     vis_results(
                #           self.exp_name, self.epoch, self.writer,
                #           self.restore_transform, img, pred_map, gt_map
                #     )

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)

        self.writer.add_scalar("val_loss", loss, self.epoch + 1)
        self.writer.add_scalar("mae", mae, self.epoch + 1)
        self.writer.add_scalar("mse", mse, self.epoch + 1)

        self.train_record = update_model(
            self.net,
            self.epoch,
            self.exp_path,
            self.exp_name,
            [mae, mse, loss],
            self.train_record,
            self.log_txt,
        )

        print_GCC_summary(
            self.log_txt,
            self.epoch,
            [mae, mse, loss],
            self.train_record,
            c_maes,
            c_mses,
        )
