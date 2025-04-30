import numpy as np
import pickle
from omegaconf import OmegaConf
from tqdm  import tqdm
from metric.features.kinetic import extract_kinetic_features
from metric.features.manual_new import extract_manual_features
from scipy import linalg
# kinetic, manual
import torch
import os, sys
import argparse
# from render import ax_to_6v
sys.path.append(os.getcwd())
from dld.data.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk


smplx_model = SMPLX_Skeleton(Jpath='data/smplx_neu_J_1.npy')

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def normalize_one(feat):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10)

def quantized_metrics(pred_features_k, gt_freatures_k, pred_features_m, gt_freatures_m):
    # pred_k_files = os.listdir(pred_k_root)
    # gt_k_files = os.listdir(gt_k_root)
    # pred_m_files = os.listdir(pred_m_root)
    # gt_m_files = os.listdir(gt_m_root)
    
    # assert len(pred_k_files) == len(gt_k_files)
    # assert len(pred_m_files) == len(gt_m_files)

    # pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    # pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'manual_features_new'))]
    
    # gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    # gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'manual_features_new'))]
    
    # pred_features_k = [np.load(os.path.join(pred_k_root, pkl)) for pkl in pred_k_files]
    # pred_features_m = [np.load(os.path.join(pred_m_root, pkl)) for pkl in pred_m_files]
    # gt_freatures_k = [np.load(os.path.join(gt_k_root, pkl)) for pkl in gt_k_files]
    # gt_freatures_m = [np.load(os.path.join(gt_m_root, pkl)) for pkl in gt_m_files]
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 
    if gt_freatures_k.shape[1] == 72:
        gt_freatures_k = gt_freatures_k[:,:66]
    if pred_features_k.shape[1] == 72:
        pred_features_k = pred_features_k[:,:66]

# T x 24 x 3 --> 72
# T x72 -->32 
    # print(gt_freatures_k.mean(axis=0))
    # print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    # print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    # print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    # print(pred_features_m.std(axis=0))

    # gt_freatures_k = normalize_one(gt_freatures_k)
    # gt_freatures_m = normalize_one(gt_freatures_m) 
    # pred_features_k = normalize_one(pred_features_k)
    # pred_features_m = normalize_one(pred_features_m)     
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)
    
    # print(gt_freatures_k.mean(axis=0))
    print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    print(pred_features_m.std(axis=0))

    
    # print(gt_freatures_k)
    # print(gt_freatures_m)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)
    # div_k_gt = '***'
    # div_m_gt = '***'
    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m' : div_m, 'div_k_gt': div_k_gt, 'div_m_gt': div_m_gt}
    return metrics


def calc_fid(kps_gen, kps_gt):

    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)
    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)
    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(root, test_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
    for file in tqdm(os.listdir(root)):
        if os.path.isdir(os.path.join(root, file)):
            continue
      
        if file[-3:] == 'npy' and file in test_list:
            if file[0] == 'M':
                continue
            data = np.load(os.path.join(root, file))
            assert len(data.shape) == 2
            if data.shape[1] == 139 or data.shape[1] == 319:
                data = data[:1024,:139]
                data = torch.from_numpy(data).to(device)   
            elif data.shape[1] == 135 or data.shape[1] == 315:
                data = data[:1024,:135]
                data = torch.from_numpy(data).to(device)   
                data = torch.cat([torch.zeros([data.shape[0], 4]).to(data),  data], dim=1) 
            assert data.shape[-1] == 139
            
            with torch.no_grad():
                joint3d = do_smplxfk(data, smplx_model)[:,:24,:]
        else:
            continue
        print(file)
        joint3d = joint3d[:1024,:22,:]
        assert len(joint3d.shape) == 3
        joint3d = joint3d.reshape(joint3d.shape[0], 22*3).detach().cpu().numpy()
            
        
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        joint3d = joint3d - np.tile(roott, (1, 22))  # Calculate relative offset with respect to root

        # relative
        joint3d_relative = joint3d.copy()
        joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
        joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]
        np.save(os.path.join(root, 'kinetic_features', file), extract_kinetic_features(joint3d_relative.reshape(-1, 22, 3)))
        np.save(os.path.join(root, 'manual_features_new', file), extract_manual_features(joint3d_relative.reshape(-1, 22, 3)))

        # np.save(os.path.join(root, 'kinetic_features', file), extract_kinetic_features(joint3d.reshape(-1, 22, 3)))
        # np.save(os.path.join(root, 'manual_features_new', file), extract_manual_features(joint3d.reshape(-1, 22, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg_assets', type=str, default='./configs/data/assets.yaml')
    parser.add_argument('--gt_root', type=str, default='./data/finedance-ballet/mofea319', help='Path to the ground truth data')
    parser.add_argument('--pred_root', type=str, default='./experiments/Local_Module/demo--ballet_base/samples_dod_inpaint_soft_ddim_2025-04-29-20-13-24/concat/npy', help='Path to the predicted data')
    
    args = parser.parse_args()
    
    # cfg = OmegaConf.load(args.cfg_assets)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    smplx_model = SMPLX_Skeleton(Jpath='data/smplx_neu_J_1.npy')

    pred_root = args.pred_root
    gt_root = args.gt_root
    
    files = os.listdir(pred_root)
    files = [f for f in files if f.endswith('.npy')]
    
    test_list = files
            
    print('test_list', test_list)
    
    calc_and_save_feats(gt_root, test_list)
    calc_and_save_feats(pred_root, test_list)
    
    print('Calculating metrics')
    print("gt_root", gt_root)
    print("pred_root", pred_root)
    
    # load files
    
    pred_features_k = [np.load(os.path.join(pred_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(pred_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(pred_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(pred_root, 'manual_features_new'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(gt_root, 'manual_features_new'))]
    
    print(quantized_metrics(pred_features_k, gt_freatures_k, pred_features_m, gt_freatures_m))
    print("gt_root", gt_root)
    print("pred_root", pred_root)
