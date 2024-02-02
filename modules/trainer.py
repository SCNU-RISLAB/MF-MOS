#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import imp
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import __init__ as booger


import torch.optim as optim
from tensorboardX import SummaryWriter as Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import warmupLR

# from modules.SalsaNextWithMotionAttention import SalsaNextWithMotionAttention
from modules.MFMOS import MFMOS

from modules.loss.Lovasz_Softmax import Lovasz_softmax, Lovasz_softmax_PointCloud
from modules.tools import AverageMeter, iouEval, save_checkpoint, show_scans_in_training, save_to_txtlog, make_log_img

from torch import distributed as dist

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, point_refine=False, local_rank=0):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.path = path
        self.epoch = 0
        self.point_refine = point_refine
        self.local_rank = local_rank

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()

        # put logger where it belongs
        self.tb_logger = Logger(self.logdir + "/tb")
        self.info = {"train_update": 0,
                     "train_loss": 0, "train_acc": 0, "train_iou": 0,
                     "valid_loss": 0, "valid_acc": 0, "valid_iou": 0,
                     "best_train_iou": 0, "best_val_iou": 0}

        # get the data
        parserModule = imp.load_source("parserModule",
                                       f"{booger.TRAIN_PATH}/common/dataset/{self.DATA['name']}/parser.py")
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          split='train',
                                          labels=self.DATA["labels"],
                                          residual_aug=self.ARCH["train"]["residual_aug"],
                                          valid_residual_delta_t=self.ARCH["train"]["valid_residual_delta_t"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["moving_learning_map"],
                                          movable_learning_map=self.DATA["movable_learning_map"],
                                          learning_map_inv=self.DATA["moving_learning_map_inv"],
                                          movable_learning_map_inv=self.DATA["movable_learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True)

        self.set_loss_weight()

        with torch.no_grad():
            self.model = MFMOS(nclasses=self.parser.get_n_classes(),
                               movable_nclasses=self.parser.get_n_classes(movable=True), params=self.ARCH)

        self.set_gpu_cuda()
        self.set_loss_function(point_refine)
        self.set_optim_scheduler()

        # if need load the pre-trained model from checkpoint
        if self.path is not None:
            self.load_pretrained_model()

    def set_loss_weight(self):
        """
            Used to calculate the weights for each class
            weights for loss (and bias)
        """
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        movable_content = torch.zeros(self.parser.get_n_classes(movable=True), dtype=torch.float)
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)   # map actual class to xentropy class
            content[x_cl] += freq

            movable_x_cl = self.parser.to_xentropy(cl, movable=True)
            movable_content[movable_x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        self.movable_loss_w = 1 / (movable_content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):   # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:    # don't weigh
                self.loss_w[x_cl] = 0

        for x_cl, w in enumerate(self.movable_loss_w):   # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:    # don't weigh
                self.movable_loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)
        print("Movable Loss weights from content: ", self.movable_loss_w.data)

    def set_loss_function(self, point_refine):
        """
            Used to define the loss function, multiple gpus need to be parallel
            # self.dice = DiceLoss().to(self.device)
            # self.dice = nn.DataParallel(self.dice).cuda()
        """
        self.criterion = nn.NLLLoss(weight=self.loss_w.double()).cuda()
        self.movable_criterion = nn.NLLLoss(weight=self.movable_loss_w.double()).cuda()
        if not point_refine:
            # self.ls = Lovasz_softmax(ignore=0).to(self.device)
            self.ls = Lovasz_softmax(ignore=0).cuda()
            self.movable_ls = Lovasz_softmax(ignore=0).cuda()
        else:
            self.ls = Lovasz_softmax_PointCloud(ignore=0).to(self.device)

        # loss as dataparallel too (more images in batch)
        # if self.n_gpus > 1:
            # self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            # self.movable_criterion = nn.DataParallel(self.movable_criterion).cuda()  # spread in gpus
            # self.ls = nn.DataParallel(self.ls).cuda()
            # print()

    def set_gpu_cuda(self):
        """
            Used to set gpus and cuda information
        """
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # cudnn.benchmark = True
            # cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            # self.model = nn.DataParallel(self.model)      # spread in gpus
            self.model = convert_model(self.model).cuda() # sync batchnorm
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=True,
                                                                   broadcast_buffers=False)

            self.model_single = self.model.module         # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()


    def set_optim_scheduler(self):
        """
            Used to set the optimizer and scheduler
        """
        self.optimizer = optim.SGD([{'params': self.model.parameters()}],
                                   lr=self.ARCH["train"]["lr"],
                                   momentum=self.ARCH["train"]["momentum"],
                                   weight_decay=self.ARCH["train"]["w_decay"])

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(optimizer=self.optimizer,
                                  lr=self.ARCH["train"]["lr"],
                                  warmup_steps=up_steps,
                                  momentum=self.ARCH["train"]["momentum"],
                                  decay=final_decay)

    def load_pretrained_model(self):
        """
            If you want to resume training, reload the model
        """
        torch.nn.Module.dump_patches = True
        if not self.point_refine:
            checkpoint = "MFMOS"
            w_dict = torch.load(f"{self.path}/{checkpoint}", map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.optimizer.load_state_dict(w_dict['optimizer'])
            self.epoch = w_dict['epoch'] + 1
            self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
            self.info = w_dict['info']
            print("info", w_dict['info'])
            print("load the pretrained model of MFMOS")
        else:
            checkpoint = "MFMOS_valid_best"
            w_dict = torch.load(f"{self.path}/{checkpoint}", map_location=lambda storage, loc: storage)
            # self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.model.load_state_dict({k.replace('module.',''):v for k,v in w_dict['state_dict'].items()})
            self.optimizer.load_state_dict(w_dict['optimizer'])
            print("load the coarse model of MFMOS_valid_best")


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) *
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                           iter + 1 + epoch * self.parser.get_train_size()))) + \
            int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def save_to_tensorboard(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            if 'valid_classes' in tag:
                continue # solve the bug of saving tensor type of value
            logger.add_scalar(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def init_evaluator(self):
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        # self.evaluator = iouEval(self.parser.get_n_classes(),
        #                          self.device, self.ignore_class)
        # self.movable_evaluator = iouEval(self.parser.get_n_classes(),
        #                                  self.device, self.ignore_class)
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.local_rank, self.ignore_class)
        self.movable_evaluator = iouEval(self.parser.get_n_classes(movable=True),
                                         self.local_rank, self.ignore_class)

    def train(self):

        self.init_evaluator()

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            self.parser.train_sampler.set_epoch(epoch)
            # train for 1 epoch
            acc, iou, loss, update_mean, hetero_l = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                                     model=self.model,
                                                                     all_criterion=(self.criterion, self.movable_criterion),
                                                                     optimizer=self.optimizer,
                                                                     epoch=epoch,
                                                                     all_evaluator=(self.evaluator, self.movable_evaluator),
                                                                     scheduler=self.scheduler,
                                                                     color_fn=self.parser.to_color,
                                                                     report=self.ARCH["train"]["report_batch"],
                                                                     show_scans=self.ARCH["train"]["show_scans"])

            if self.local_rank == 0:
                # update the info dict and save the training checkpoint
                self.update_training_info(epoch, acc, iou, loss, update_mean, hetero_l)

            # evaluate on validation set
            if epoch % self.ARCH["train"]["report_epoch"] == 0 and self.local_rank == 0:
                acc, iou, loss, rand_img, hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                                   model=self.model,
                                                                   all_criterion=(self.criterion, self.movable_criterion),
                                                                   all_evaluator=(self.evaluator, self.movable_evaluator),
                                                                   class_func=self.parser.get_xentropy_class_string,
                                                                   color_fn=self.parser.to_color,
                                                                   save_scans=self.ARCH["train"]["save_scans"])

                self.update_validation_info(epoch, acc, iou, loss, hetero_l)

            if self.local_rank == 0:
                # save to tensorboard log
                Trainer.save_to_tensorboard(logdir=self.logdir,
                                            logger=self.tb_logger,
                                            info=self.info,
                                            epoch=epoch,
                                            w_summary=self.ARCH["train"]["save_summary"],
                                            model=self.model_single,
                                            img_summary=self.ARCH["train"]["save_scans"],
                                            imgs=rand_img)

            dist.barrier()

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, all_criterion, optimizer,
                    epoch, all_evaluator, scheduler, color_fn, report=10,
                    show_scans=False):

        losses = AverageMeter()

        moving_losses = AverageMeter()
        movable_losses = AverageMeter()

        acc = AverageMeter()
        iou = AverageMeter()

        movable_acc = AverageMeter()
        movable_iou = AverageMeter()

        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()

        evaluator, movable_evaluator = all_evaluator
        criterion, movable_criterion = all_criterion

        # empty the cache to train now
        # if self.gpu:
        #     torch.cuda.empty_cache()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (in_vol, proj_mask, all_proj_labels, _, path_seq, path_name,
                _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
            # measure data loading time
            self.data_time_t.update(time.time() - end)

            proj_labels, movable_proj_labels = all_proj_labels

            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()
                movable_proj_labels = movable_proj_labels.cuda().long()

            output, _, movable_output, _ = model(in_vol)

            # loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())
            moving_loss_m = criterion(torch.log(output.clamp(min=1e-8)).double(), proj_labels).float() + self.ls(output, proj_labels.long())
            movable_loss_m = movable_criterion(torch.log(movable_output.clamp(min=1e-8)).double(), movable_proj_labels).float() \
                             + self.movable_ls(movable_output, movable_proj_labels.long())

            loss_m = moving_loss_m + movable_loss_m

            optimizer.zero_grad()
            # if self.n_gpus > 1:
            #     idx = torch.ones(self.n_gpus).cuda()
            #     loss_m.backward(idx)
            # else:
            #     loss_m.backward()
            loss_m.backward()
            optimizer.step()

            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

                movable_evaluator.reset()
                movable_argmax = movable_output.argmax(dim=1)
                movable_evaluator.addBatch(movable_argmax, movable_proj_labels)
                movable_accuracy = movable_evaluator.getacc()
                movable_jaccard, movable_class_jaccard = movable_evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            moving_losses.update(moving_loss_m.mean().item(), in_vol.size(0))
            movable_losses.update(movable_loss_m.mean().item(), in_vol.size(0))

            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            movable_acc.update(movable_accuracy.item(), in_vol.size(0))
            movable_iou.update(movable_jaccard.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) * value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                show_scans_in_training(
                    proj_mask, in_vol, argmax, proj_labels, color_fn)
                show_scans_in_training(
                    proj_mask, in_vol, movable_argmax, movable_proj_labels, color_fn, movable=True)

            if i % self.ARCH["train"]["report_batch"] == 0 and self.local_rank == 0:
                str_line = ('Lr: {lr:.3e} | '
                            'Update: {umean:.3e} mean,{ustd:.3e} std | '
                            'Epoch: [{0}][{1}/{2}] | '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                            'MovingLoss {moving_losses.val:.4f} ({moving_losses.avg:.4f}) | '
                            'MovableLoss {movable_losses.val:.4f} ({movable_losses.avg:.4f}) | '
                            'MovingAcc {moving_acc.val:.3f} ({moving_acc.avg:.3f}) | '
                            'MovingIoU {moving_iou.val:.3f} ({moving_iou.avg:.3f}) | '
                            'movable_acc {movable_acc.val:.3f} ({movable_acc.avg:.3f}) | '
                            'movable_IoU {movable_iou.val:.3f} ({movable_iou.avg:.3f}) | '
                            '[{estim}]').format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t, data_time=self.data_time_t, loss=losses,
                    moving_losses=moving_losses, movable_losses=movable_losses,
                    moving_acc=acc, moving_iou=iou,
                    movable_acc=movable_acc, movable_iou=movable_iou,
                    lr=lr, umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i))
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg, hetero_l.avg

    def validate(self, val_loader, model, all_criterion, all_evaluator, class_func, color_fn, save_scans=False):
        evaluator, movable_evaluator = all_evaluator
        criterion, movable_criterion = all_criterion

        losses = AverageMeter()
        moving_losses = AverageMeter()
        movable_losses = AverageMeter()

        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        movable_jaccs = AverageMeter()
        movable_wces = AverageMeter()
        movable_acc = AverageMeter()
        movable_iou = AverageMeter()

        hetero_l = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()
        movable_evaluator.reset()

        # empty the cache to infer in high res
        # if self.gpu:
        #     torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, all_proj_labels, _, path_seq, path_name,
                    _, _, _, _, _, _, _, _, _)\
                    in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):
                proj_labels, movable_proj_labels = all_proj_labels

                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()
                    movable_proj_labels = movable_proj_labels.cuda(non_blocking=True).long()

                # compute output
                output, _, movable_output, _ = model(in_vol)

                log_out = torch.log(output.clamp(min=1e-8))
                movable_log_out = torch.log(movable_output.clamp(min=1e-8))

                # wce = criterion(log_out, proj_labels)
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out.double(), proj_labels).float()
                moving_loss = wce + jacc

                movable_jacc = self.ls(movable_output, movable_proj_labels)
                movable_wce = movable_criterion(movable_log_out.double(), movable_proj_labels).float()
                movable_loss = movable_wce + movable_jacc

                loss = moving_loss + movable_loss

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)

                movable_argmax = movable_output.argmax(dim=1)
                movable_evaluator.addBatch(movable_argmax, movable_proj_labels)

                losses.update(loss.mean().item(), in_vol.size(0))

                moving_losses.update(moving_loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))
                wces.update(wce.mean().item(), in_vol.size(0))

                movable_losses.update(movable_loss.mean().item(), in_vol.size(0))
                movable_jaccs.update(movable_jacc.mean().item(), in_vol.size(0))
                movable_wces.update(movable_wce.mean().item(), in_vol.size(0))


                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            movable_accuracy = movable_evaluator.getacc()
            movable_jaccard, movable_class_jaccard = movable_evaluator.getIoU()
            movable_acc.update(movable_accuracy.item(), in_vol.size(0))
            movable_iou.update(movable_jaccard.item(), in_vol.size(0))

            str_line = ("*" * 80 + '\n'
                        'Validation set:\n'
                        'Time avg per batch {batch_time.avg:.3f}\n'
                        'Loss avg {loss.avg:.4f}\n'
                        'MovingLoss avg {moving_loss.avg:.4f}\n'
                        'MovableLoss avg {movable_loss.avg:.4f}\n'
                        'MovingJaccard avg {moving_jac.avg:.4f}\n'
                        'MovingWCE avg {moving_wces.avg:.4f}\n'
                        'MovingAcc avg {moving_acc.avg:.6f}\n'
                        'MovingIoU avg {moving_iou.avg:.6f}\n'
                        'MovableJaccard avg {movable_jac.avg:.4f}\n'
                        'MovableWCE avg {movable_wce.avg:.4f}\n'
                        'MovableAcc avg {movable_acc.avg:.6f}\n'
                        'MovableIoU avg {movable_iou.avg:.6f}'
                        ).format(
                            batch_time=self.batch_time_e, loss=losses,
                            moving_loss=moving_losses, movable_loss=movable_losses,
                            moving_jac=jaccs, moving_wces=wces, moving_acc=acc, moving_iou=iou,
                            movable_jac=movable_jaccs, movable_wce=movable_wces,
                            movable_acc=movable_acc, movable_iou=movable_iou)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)

            print("-" * 80)
            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc
                str_line = 'IoU class {i:} [{class_str:}] = {jacc:.6f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc)
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

            print("-" * 80)
            for i, jacc in enumerate(movable_class_jaccard):
                self.info["valid_classes/" + class_func(i, movable=True)] = jacc
                str_line = 'IoU class {i:} [{class_str:}] = {jacc:.6f}'.format(
                    i=i, class_str=class_func(i, movable=True), jacc=jacc)
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

            str_line = '*' * 80
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)

        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg

    def update_training_info(self, epoch, acc, iou, loss, update_mean, hetero_l):
        # update info
        self.info["train_update"] = update_mean
        self.info["train_loss"] = loss
        self.info["train_acc"] = acc
        self.info["train_iou"] = iou
        self.info["train_hetero"] = hetero_l

        # remember best iou and save checkpoint
        state = {'epoch': epoch,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'info': self.info,
                 'scheduler': self.scheduler.state_dict()}
        save_checkpoint(state, self.logdir, suffix="")

        if self.info['train_iou'] > self.info['best_train_iou']:
            print("Best mean iou in training set so far, save model!")
            self.info['best_train_iou'] = self.info['train_iou']
            state = {'epoch': epoch,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()}
            save_checkpoint(state, self.logdir, suffix="_train_best")

    def update_validation_info(self, epoch, acc, iou, loss, hetero_l):
        # update info
        self.info["valid_loss"] = loss
        self.info["valid_acc"] = acc
        self.info["valid_iou"] = iou
        self.info['valid_heteros'] = hetero_l

        # remember best iou and save checkpoint
        if self.info['valid_iou'] > self.info['best_val_iou']:
            str_line = ("Best mean iou in validation so far, save model!\n" + "*" * 80)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)
            self.info['best_val_iou'] = self.info['valid_iou']

            # save the weights!
            state = {'epoch': epoch,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()}
            save_checkpoint(state, self.logdir, suffix="_valid_best")
            save_checkpoint(state, self.logdir, suffix=f"_valid_best_{epoch}")

