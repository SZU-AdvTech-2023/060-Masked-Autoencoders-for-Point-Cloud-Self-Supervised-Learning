import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from datasets.FRGCLoader import FRGC_train_pair,synthetic_train
from datasets.KinectLowQLoader import KinectLQ_train,KinectLQ_eval
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
import datasets.data_utils as d_utils
import torch.optim as optim
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torch.utils.data import DataLoader
from models.Point_MAE import Discriminator
from gradient_penalty import GradientPenalty
train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    # (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
    #                                                         builder.dataset_builder(args, config.dataset.val)
    # (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    # train_dataset = FRGC_train_pair(num_points=4096, root="/data2/gaoziqi/Point-MAE-main/data", transforms=None, train=True)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=config.total_bs,
    #     shuffle=True,
    #     num_workers = int(args.num_workers),
    #     drop_last=False
    # )
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    #train_dataset = KinectLQ_train(num_points=4096, root="/data2/gaoziqi/PointFace/dataset", transforms=train_transforms, train=True)
    #train_dataset = FRGC_train_pair(num_points=4096, root="/data2/gaoziqi/Point-MAE-main/data", transforms=train_transforms, train=True)
    train_dataset=synthetic_train(num_points=4096, root="/data1/gaoziqi/3DFaceMAE-copy/data", transforms=train_transforms, train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.total_bs,
        shuffle=True,
        num_workers = 16,
        drop_last=False
    )
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    best_losses=10
    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    D_network = nn.DataParallel(Discriminator(batch_size=config.total_bs, features=args.D_FEAT)).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GP = GradientPenalty(10, gamma=1,device=device)
    #if args.resume:
        #builder.resume_optimizer(optimizer, args, logger = logger)
    optimizerD,schedulerD = builder.build_opti_sche(D_network, config)
    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        losses2=AverageMeter(['Loss_total'])
        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (data,labels) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
           # print(npoints)
            #print(points.size(1))
            #center=center.cuda()
            #points=points[...,:3].contiguous()
            
            # D_network.zero_grad()
            # real_point=torch.stack([points[i] for i in range(labels.shape[0]) if labels[i]==1])
            # D_real, real_index = D_network(real_point)
            # D_realm = D_real.mean()
            # fake_point=torch.stack([points[i] for i in range(labels.shape[0]) if labels[i]==0])
            # D_fake, _ = D_network(fake_point)
            # D_fakem = D_fake.mean()
                    #print(point.data.size())
                    #print(fake_point.data.size())
            #gp_loss = GP(D_network, real_point, fake_point)
                    
            # d_loss = -D_realm + D_fakem
            # #d_loss_gp = d_loss + gp_loss
            # d_loss.backward()
            # optimizerD.step()
            assert points.size(1) == npoints
            #points = train_transforms(points)
            loss1,loss2 = base_model(points)
            loss=loss1+loss2
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss1.mean().item()*1000])
                losses2.update([loss.item()*1000])
            else:
                losses.update([loss1.mean().item()*1000])
                losses2.update([loss.item()*1000])

            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            #avg_losses=losses.avg()[0]
            # print(avg_losses)
            # if avg_losses<best_losses:
            #         print(1)
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) total_loss= %s Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),['%.4f' % l for l in losses2.val()],
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
                # avg_losses=torch.stack(losses.avg())
                # print(avg_losses)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
            for item in schedulerD:
                item.step(epoch)
        else:
            scheduler.step(epoch)
            schedulerD.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
            # Save ckeckpoints
        avg_losses=losses.avg()[0]#.mean()
        if avg_losses<best_losses:
                best_losses =avg_losses
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=100:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass