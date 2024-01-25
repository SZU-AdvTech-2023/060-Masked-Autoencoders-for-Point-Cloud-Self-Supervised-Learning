import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from datasets.KinectLowQLoader import KinectLQ_train_pair,KinectLQ_eval,KinectLQ_train
from datasets.BosphLoader import Bosphorus,Bosphorus_eval
import numpy as np
from datasets.FRGCLoader import FRGC_train_pair
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from torch.utils.data import DataLoader
import math
from torch.autograd import Variable
import torch.nn.functional as F
from ContrastiveLoss import MultiContrastiveLoss
import datasets.data_utils as d_utils
train_transforms = transforms.Compose(
    [
         # data_transforms.PointcloudScale(),
         # data_transforms.PointcloudRotate(),
         # data_transforms.PointcloudTranslate(),
         # data_transforms.PointcloudJitter(),
         # data_transforms.PointcloudRandomInputDropout(),
         # data_transforms.RandomHorizontalFlip(),
         data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)
def data_augmentation(point_cloud):
    """point cloud data augmentation using function from data_utils.py"""

    PointcloudRandomInputDropout = d_utils.PointcloudRandomInputDropout()
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()
    PointcloudJitter = d_utils.PointcloudJitter(clip=0.02)
    # PointcloudJitter = d_utils.PointcloudJitterAxisZ(std=0.05)
    # PointcloudOcclusion = d_utils.PointcloudManmadeOcclusion()
    angle = math.pi/2.0
    
    PointcloudRotatebyRandomAngle_y = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([0.0, 1.0, 0.0]))
    PointcloudRotatebyRandomAngle_x = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([1.0, 0.0, 0.0]))
    PointcloudRotatebyRandomAngle_z = d_utils.PointcloudRotatebyRandomAngle(rotation_angle=angle, 
        axis=np.array([0.0, 0.0, 1.0]))  
  

    transform_func = {0: lambda x: x,
            1:PointcloudScaleAndTranslate,
            2:PointcloudRotatebyRandomAngle_x,
            3:PointcloudRotatebyRandomAngle_y,
            # 4:PointcloudRotatebyRandomAngle_z,
            # 4:PointcloudOcclusion
            4:PointcloudJitter,
            }

    method_num = 1

    func_id = np.array(list(transform_func.keys()))
    pro = [0.2, 0.2, 0.2, 0.2, 0.2] # probability of each transform function
    # other_pro = (1-pro[0])/float(len(transform_func)-1)
    # pro.extend([other_pro]*(func_id.shape[0]-1))

    func_use_id = np.random.choice(func_id, method_num, replace=False, p=pro)
    
    if 0 in list(func_use_id):
        return PointcloudRandomInputDropout(point_cloud)#,PointcloudRandomInputDropout(key)

    for idx in func_use_id:
        point_cloud = transform_func[idx](point_cloud)
        #key=transform_func[idx](key)
    point_cloud = PointcloudRandomInputDropout(point_cloud)
    #key=PointcloudRandomInputDropout(key)
    return point_cloud#,key


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
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

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    criterion_globfeat = MultiContrastiveLoss(margin=0.35, scale=1.0)
    # build dataset
    # (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
    #                                                         builder.dataset_builder(args, config.dataset.val)
    train_dataset = KinectLQ_train_pair(num_points=4096, root="/data1/gaoziqi/3DFaceMAE-copy/data", transforms=None, train=True,format=config.format)
    #train_dataset=Bosphorus(num_points=4096,root="/data2/gaoziqi/Point-MAE-main/data",transforms=None,train=True)
    # train_dataset = FRGC_train_pair(num_points=4096, root="/data2/gaoziqi/Point-MAE-main/data", transforms=None, train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.total_bs,
        shuffle=True,
        num_workers = 16,
        drop_last=False
    )
    test_dataset = KinectLQ_eval(num_points = 4096, root ="/data1/gaoziqi/3DFaceMAE-copy/data", transforms=None
                                 , valtxt='OC_val_du.txt', format=config.format)
    # test_dataset =Bosphorus_eval(num_points = 4096, root ="/data2/gaoziqi/Point-MAE-main/data", transforms=None
    #                            )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.total_bs,
        shuffle=False, 
        num_workers=4
    )
    gallery_points, gallery_labels = test_dataset.get_gallery()
    gallery_points, gallery_labels = gallery_points.cuda(), gallery_labels.cuda()
    gallery_points =  Variable(gallery_points)
    #gallery_keys=Variable(gallery_keys)
    #gallery_num = gallery_labels.shape[0]
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    #print(optimizer.param_groups)
    # for p in base_model.module.group_divider.parameters():
    #         p.requires_grad=True
    # optimizer.add_param_group({'params': base_model.module.group_divider.parameters()})
    #             #print(optimizer.param_groups)
    # for p in base_model.module.encoder.parameters():
    #         p.requires_grad=True
    # optimizer.add_param_group({'params': base_model.module.encoder.parameters()})
    #             #for p in base_model.module.cls_pos.parameters():
    # base_model.module.cls_pos.requires_grad=True
    # optimizer.add_param_group({'params': base_model.module.cls_pos})
    # base_model.module.cls_token.requires_grad=True
    # optimizer.add_param_group({'params': base_model.module.cls_token})
    # for p in base_model.module.pos_embed.parameters():
    #     p.requires_grad=True
    # optimizer.add_param_group({'params': base_model.module.pos_embed.parameters()})  
    # for p in base_model.module.norm.parameters():
    #     p.requires_grad=True
    # optimizer.add_param_group({'params': base_model.module.norm.parameters()})
    # for p in base_model.module.blocks.parameters():
    #     p.requires_grad=True
    # optimizer.add_param_group({'params': base_model.module.blocks.parameters()})
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    unfreezing=0
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()
        
        #metrics = validate(base_model, test_dataloader,gallery_points,gallery_keys,gallery_labels, epoch, val_writer, args, config, logger=logger)
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        #metrics = validate(base_model, test_dataloader,test_dataset, epoch, val_writer, args, config, logger=logger)
        npoints = config.npoints
        for idx, data in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            points1, target1,points2,target2= data
            points1, target1,points2,target2= points1.cuda(),target1.cuda(),points2.cuda(), target2.cuda()
            points1, target1 = Variable(points1), Variable(target1)
            points2, target2 = Variable(points2), Variable(target2)
            #key1,key2=Variable(key1),Variable(key2)
            #points1=points1[...,:3].contiguous()
            #points2=points2[...,:3].contiguous()
            #print(points1.size())
            # fastest point sampling
            # fps_idx = pointnet2_utils.furthest_point_sample(points1, args.num_points)  # (B, npoint)
            # points1 = pointnet2_utils.gather_operation(points1.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            # fps_idx = pointnet2_utils.furthest_point_sample(points2, args.num_points)  # (B, npoint)
            # points2 = pointnet2_utils.gather_operation(points2.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)            
            # augmentation
            points1 = data_augmentation(points1)
            points2 = data_augmentation(points2)          
            data_time.update(time.time() - batch_start_time)
            
            # points = data[0].cuda()
            # label = data[1].cuda()

            # if npoints == 1024:
            #     point_all = 1200
            # elif npoints == 2048:
            #     point_all = 2400
            # elif npoints == 4096:
            #     point_all = 4800
            # elif npoints == 8192:
            #     point_all = 8192
            # else:
            #     raise NotImplementedError()

            # if points.size(1) < point_all:
            #     point_all = points.size(1)

            # fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            # fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            # points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            # # import pdb; pdb.set_trace()
            # points = train_transforms(points)

            features_A, pred_A,features_B,pred_B = base_model((points1,points2))
            target1 = target1.view(-1)
            target2=target2.view(-1)
            loss_A, acc_A = base_model.module.get_loss_acc(pred_A, target1)
            loss_B, acc_B = base_model.module.get_loss_acc(pred_B,target2)
            loss_gf,dist= criterion_globfeat(features_A, features_B, target1, target2)
            #small_idx=[num_idx[i] for i in range(len(num_idx)) if dist[i]<0.4 and dist[i]<dist.mean()]
            #train_dataloader.dataset.pair_update(small_idx)
            loss=loss_A+loss_B+loss_gf
            acc=acc_A
            _loss = loss_A+loss_B+loss_gf
            # print(loss_gf)
            # print(loss_A)
            # print(loss_B)
            _loss.backward()
            #print(1)
            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()


            # if epoch ==50 and unfreezing==0:
            #     for p in base_model.module.group_divider.parameters():
            #         p.requires_grad=True
            #     optimizer.add_param_group({'params': base_model.module.group_divider.parameters()})
            #     #print(optimizer.param_groups)
            #     for p in base_model.module.encoder.parameters():
            #         p.requires_grad=True
            #     optimizer.add_param_group({'params': base_model.module.encoder.parameters()})
            #     #for p in base_model.module.cls_pos.parameters():
            #     base_model.module.cls_pos.requires_grad=True
            #     optimizer.add_param_group({'params': base_model.module.cls_pos})
            #     base_model.module.cls_token.requires_grad=True
            #     optimizer.add_param_group({'params': base_model.module.cls_token})
            #     for p in base_model.module.pos_embed.parameters():
            #         p.requires_grad=True
            #     optimizer.add_param_group({'params': base_model.module.pos_embed.parameters()})  
            #     for p in base_model.module.norm.parameters():
            #         p.requires_grad=True
            #     optimizer.add_param_group({'params': base_model.module.norm.parameters()})
            #     for p in base_model.module.blocks.parameters():
            #         p.requires_grad=True
            #     optimizer.add_param_group({'params': base_model.module.blocks.parameters()})
                #print(optimizer.param_groups)
               # unfreezing=1
                    
                    
                    
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 40 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

       # if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
        metrics = validate(base_model, test_dataloader,gallery_points,gallery_labels, epoch, val_writer, args, config, logger=logger)

        better = metrics.better_than(best_metrics)
            # Save ckeckpoints
        if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
        if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader,gallery_points,gallery_labels, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)\
    samples=0
    correct=0
    base_model.eval()  # set model to eval mode
    NUM_REPEAT = 10
    G_BATCH_SIZE=16
    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        Total_samples = 0
        Correct = 0
        # gallery_points, gallery_labels = test_dataset.get_gallery()
        # gallery_points, gallery_labels = gallery_points.cuda(), gallery_labels.cuda()
        # gallery_points =  Variable(gallery_points)

        gallery_num = gallery_labels.shape[0]
        #gallery_labels_new = torch.zeros(gallery_num//G_BATCH_SIZE, dtype=torch.long).cuda()
        for i in np.arange(0, gallery_num//G_BATCH_SIZE + 1):
            #print(gallery_points[i*G_BATCH_SIZE:i+G_BATCH_SIZE,:,:].shape)
            if i < gallery_num//G_BATCH_SIZE:
                g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:]
                #g_keys=gallery_keys[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:]
                # fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
                # g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                tmp_pred, m_b,_,_= base_model((g_points,g_points))
                # tmp_pred = model(g_points)
                if i==0:
                    gallery_pred = torch.tensor(tmp_pred).clone().cuda()
                else:
                    gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)

            if i==gallery_num//G_BATCH_SIZE:
                num_of_rest = gallery_num % G_BATCH_SIZE
                g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+num_of_rest,:,:]
                #g_keys=gallery_keys[i*G_BATCH_SIZE:i*G_BATCH_SIZE+num_of_rest,:,:]
                # fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
                # g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                tmp_pred, m_b,_,_ = base_model((g_points,g_points))
                # tmp_pred = model(g_points)
                gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)
                # print(tmp_pred.size(), gallery_pred.size())

        # gallery_labels_new = gallery_labels.clone()
        print("gallery features size:{}".format(gallery_pred.size()))
        gallery_pred = F.normalize(gallery_pred)

        for idx, data in enumerate(test_dataloader):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda() 

            probe_pred, m_b,_,_= base_model((probe_points,probe_points))
            target = probe_labels.view(-1)

            #pred = logits.argmax(-1).view(-1)

            probe_pred = F.normalize(probe_pred)
           
            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0], 
                                                        probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                        gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2) # cosine distance
            # results = torch.sum(torch.pow(probe_tmp-gallery_tmp, 2), dim=2) # l2 distance
            results = torch.argmax(results, dim=1)
            # results = torch.argmin(results, dim=1) # l2 judge
            
            Total_samples += probe_points.shape[0]
            samples+=probe_points.shape[0]
            #print(results, probe_labels)
            
            for i in np.arange(0, results.shape[0]):
                if gallery_labels[results[i]] == probe_labels[i]:
                    correct += 1
        acc = float(correct/samples)
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
     
    test(base_model, test_dataloader, args, config, logger=logger)
    
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc
