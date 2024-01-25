import h5py
import numpy as np
import os
import torch
from pointnet2_ops import pointnet2_utils
with h5py.File('/data2/gaoziqi/Point-MAE-main/model2017-1_face12_nomouth.h5',"r") as f:
    shape_mean=f['shape']['model']['mean']
    shape_mean=np.array(shape_mean)
    shape_noiseVariance=f['shape']['model']['noiseVariance']
    shape_noiseVariance=np.array(shape_noiseVariance)
    shape_pcaBasis=f['shape']['model']['pcaBasis']
    shape_pcaBasis=np.array(shape_pcaBasis)
    shape_pcaVariance=f['shape']['model']['pcaVariance']
    shape_pcaVariance=np.array(shape_pcaVariance)
    exp_mean=f['expression']['model']['mean']
    exp_mean=np.array(exp_mean)
    exp_noiseVariance=f['expression']['model']['noiseVariance']
    exp_noiseVariance=np.array(exp_noiseVariance)
    exp_pcaBasis=f['expression']['model']['pcaBasis']
    exp_pcaBasis=np.array(exp_pcaBasis)
    exp_pcaVariance=f['expression']['model']['pcaVariance']
    exp_pcaVariance=np.array(exp_pcaVariance)
shapes=10000
expressions=25
save_folder='/data2/gaoziqi/Point-MAE-main/data/syn_Data/'
if os.path.exists(save_folder)!=1:
    os.mkdir(save_folder)
ClassNameFromat=400000000
mean_alfa=np.zeros((199,1))
mean_beta=np.zeros((100,1))
Alfa=np.concatenate((mean_alfa,np.random.randn(199,shapes)),axis=1)
Beta=np.concatenate((mean_beta,np.random.randn(100,shapes)),axis=1)
#print(Beta)
k=0
files_num=1
fake_pointclouds = torch.Tensor([])
for i in range(1,shapes+1):
    alfa = Alfa[:,i]
    
    ShapeFolder = save_folder
    ScanNameFormat=000
    for j in range(1,expressions+1):
        print('class [%d]/[%d]: expression[%d]/[%d]\n'%(i,shapes,j,expressions))
        beta = Beta[:,j]
        #print(exp_pcaBasis.shape)
        exp_beta = np.dot(exp_pcaBasis, (beta * np.sqrt(exp_pcaVariance)))
        shape_alfa = np.dot(shape_pcaBasis ,(alfa * np.sqrt(shape_pcaVariance)))
        #print(exp_beta.shape)
        face = shape_mean + shape_alfa + exp_mean + exp_beta
        #np.savetxt(save_folder+"1.txt",face)       
        face=face.reshape(3,len(face)//3,order="F").T
        
        dsface=face[8157,:]
        face = face - face[8157,:]
        choice = np.random.choice(len(face), len(face), replace=False)       
        sample = face[choice, :]#choice
        
        sample[:,0:3] = (sample[:,0:3])/(100)
        ScanName=os.path.join(ShapeFolder,str(k))
        # fps_idx = pointnet2_utils.furthest_point_sample(torch.tensor(sample).float().cuda(), 10000)  # (B, npoint)
        # sample = pointnet2_utils.gather_operation(torch.tensor(sample).float().cuda().unsqueeze(0).transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  
        #print(ScanName)
        np.savetxt(ScanName+".txt",sample,fmt="%.3f %.3f %.3f")
        k=k+1
    #dwq=print(torch.tensor(np.array(pcd.points)).float().type())
        #print(j)
        #fake_pointclouds = torch.cat((fake_pointclouds, torch.tensor(sample)), dim=0)
    #np.savetxt("C:/Users/dream/Downloads/Compressed/3DFacePointCloudNet-master/3DFacePointCloudNet-master/Result/"+j,sample)
    # if i%1000==0:
    #     np.save("./data/generated_face"+"_"+str(files_num)+".npy",fake_pointclouds.cpu().numpy())
    #     fake_pointclouds=torch.tensor([])
    #     files_num=files_num+1