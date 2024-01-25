import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from datasets.KinectLowQLoader import KinectLQ_train_pair,KinectLQ_eval
from datasets.BosphLoader import Bosphorus_eval
from torch.utils.data import DataLoader
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    test_dataset = KinectLQ_eval(num_points = 4096, root ="/data1/gaoziqi/3DFaceMAE-copy/data", transforms=None
                                , valtxt='OC_val_du.txt', format=config.format)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.total_bs,
        shuffle=False, 
        num_workers=8
    )
    test_dataset_2 = KinectLQ_eval(num_points =4096, root = "/data1/gaoziqi/3DFaceMAE-copy/data", transforms=None
                                , valtxt='TM_val_du.txt', format=config.format)
    test_dataloader_2 = DataLoader(
        test_dataset_2, 
        batch_size=config.total_bs,
        shuffle=False, 
        num_workers=8
    )
    print("1")
    test_dataset_3 = KinectLQ_eval(num_points = 4096, root = "/data1/gaoziqi/3DFaceMAE-copy/data", transforms=None
                                , valtxt='PS_val_du.txt', format=config.format)
    test_dataloader_3 = DataLoader(
        test_dataset_3, 
        batch_size=config.total_bs,
        shuffle=False, 
        num_workers=8
    )
    print("1")
    test_dataset_4 = KinectLQ_eval(num_points = 4096, root = "/data1/gaoziqi/3DFaceMAE-copy/data", transforms=None
                                , valtxt='FE_val_du.txt', format=config.format)
    test_dataloader_4 = DataLoader(
        test_dataset_4, 
        batch_size=config.total_bs,
        shuffle=False, 
        num_workers=8
    )
    print("1")
    test_dataset_5 = KinectLQ_eval(num_points = 4096, root = "/data1/gaoziqi/3DFaceMAE-copy/data", transforms=None
                                , valtxt='new_NU_val_du.txt', format=config.format)
    test_dataloader_5 = DataLoader(
        test_dataset_5, 
        batch_size=config.total_bs,
        shuffle=False, 
        num_workers=8
    )
    print("1")
    # test_dataset =Bosphorus_eval(num_points = 4096, root ="/data2/gaoziqi/Point-MAE-main/data", transforms=None
    #                            )
    # test_dataloader = DataLoader(
    #     test_dataset, 
    #     batch_size=config.total_bs,
    #     shuffle=False, 
    #     num_workers=4
    # )
    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, (test_dataloader,test_dataloader_2,test_dataloader_3,test_dataloader_4,test_dataloader_5),test_dataset, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader,test_dataset, args, config, logger = None):
    #test_dataloader
    test_dataloader,test_dataloader_2,test_dataloader_3,test_dataloader_4,test_dataloader_5=test_dataloader
    base_model.eval()  # set model to eval mode
    target = './vis'
    # useful_cate = [
    #     "02691156", #plane
    #     "04379243",  #table
    #     "03790512", #motorbike
    #     "03948459", #pistol
    #     "03642806", #laptop
    #     "03467517",     #guitar
    #     "03261776", #earphone
    #     "03001627", #chair
    #     "02958343", #car
    #     "04090263", #rifle
    #     "03759954", # microphone
    # ]
    with torch.no_grad():
        Total_samples = 0
        #Correct = 0
        samples=0
        correct_1=0
        NUM_REPEAT = 10
        G_BATCH_SIZE=16
        gallery_points,gallery_labels = test_dataset.get_gallery()
        gallery_points, gallery_labels = gallery_points.cuda(), gallery_labels.cuda()
        gallery_points =  Variable(gallery_points)

        gallery_num = gallery_labels.shape[0]
        #gallery_labels_new = torch.zeros(gallery_num//G_BATCH_SIZE, dtype=torch.long).cuda()
        for i in np.arange(0, gallery_num//G_BATCH_SIZE + 1):
            #print(gallery_points[i*G_BATCH_SIZE:i+G_BATCH_SIZE,:,:].shape)
            if i < gallery_num//G_BATCH_SIZE:
                g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+G_BATCH_SIZE,:,:]
                # fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
                # g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                tmp_pred, m_b = base_model((g_points, g_points))
                # tmp_pred = model(g_points)
                if i==0:
                    gallery_pred = torch.tensor(tmp_pred).clone().cuda()
                else:
                    gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)

            if i==gallery_num//G_BATCH_SIZE:
                num_of_rest = gallery_num % G_BATCH_SIZE
                g_points = gallery_points[i*G_BATCH_SIZE:i*G_BATCH_SIZE+num_of_rest,:,:]
                # fps_idx = pointnet2_utils.furthest_point_sample(g_points, args.num_points)  # (B, npoint)
                # g_points = pointnet2_utils.gather_operation(g_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                tmp_pred, m_b= base_model((g_points, g_points))
                # tmp_pred = model(g_points)
                gallery_pred = torch.cat((gallery_pred, tmp_pred), dim=0)
                # print(tmp_pred.size(), gallery_pred.size())

        # gallery_labels_new = gallery_labels.clone()
        print("gallery features size:{}".format(gallery_pred.size()))
        gallery_pred = F.normalize(gallery_pred)

        for idx, data in enumerate(tqdm(test_dataloader)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda() 

            probe_pred, m_b = base_model((probe_points,probe_points))
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
                    correct_1 += 1
        print_log('Eval Set 1 total_samples:{} correct:{}'.format(samples,correct_1),logger=logger)      
        acc = float(correct_1/samples)
        print_log('\n test_dataset_1 {} acc: {:.6f}\n'.format("OC", acc),logger=logger)
        correct_2=0
        samples=0
        for idx, data in enumerate(tqdm(test_dataloader_2)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda() 

            probe_pred, m_b = base_model((probe_points,probe_points))
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
                    correct_2 += 1
        print_log('Eval Set 2 total_samples:{} correct:{}'.format(samples,correct_2),logger=logger)      
        acc = float(correct_2/samples)
        print_log('\n test_dataset_2 {} acc: {:.6f}\n'.format("TM", acc),logger=logger)
        correct_3=0
        samples=0
        for idx, data in enumerate(tqdm(test_dataloader_3)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda() 

            probe_pred, m_b = base_model((probe_points,probe_points))
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
                    correct_3 += 1
        print_log('Eval Set 3 total_samples:{} correct:{}'.format(samples,correct_3),logger=logger)      
        acc = float(correct_3/samples)
        print_log('\n test_dataset_3 {} acc: {:.6f}\n'.format("PS", acc),logger=logger)
        correct_4=0
        samples=0
        for idx, data in enumerate(tqdm(test_dataloader_4)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda() 

            probe_pred, m_b = base_model((probe_points,probe_points))
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
                    correct_4 += 1
        print_log('Eval Set 4 total_samples:{} correct:{}'.format(samples,correct_4),logger=logger)      
        acc = float(correct_4/samples)
        print_log('\n test_dataset_4 {} acc: {:.6f}\n'.format("FE", acc),logger=logger)
        samples=0
        correct_5=0
        for idx, data in enumerate(tqdm(test_dataloader_5)):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda() 

            probe_pred, m_b = base_model((probe_points,probe_points))
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
                    correct_5 += 1
        print_log('Eval Set 5 total_samples:{} correct:{}'.format(samples,correct_5),logger=logger)      
        acc = float(correct_5/samples)
        print_log('\n test_dataset_5 {} acc: {:.6f}\n'.format("NU", acc),logger=logger)
        Correct=correct_1+correct_2+correct_3+correct_4+correct_5
        print_log('Eval Set total_samples:{}'.format(Total_samples),logger=logger)      
        acc = float(Correct/Total_samples)
        print_log('\n total acc: {:.6f}\n'.format(acc),logger=logger)
        #acc = float(correct/samples)
        #print(acc)
            # if idx > 1500:
            #     brea
        return
