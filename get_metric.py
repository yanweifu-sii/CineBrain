import os, sys
import numpy as np
from models.eval_metrics import (clip_score_only, 
                          ssim_score_only, 
                          img_classify_metric, 
                          video_classify_metric,
                          remove_overlap)
import imageio.v3 as iio
import torch

def main(
        data_path
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    gt_list = []
    pred_list = []
    print('loading test results ...')
    for i in range(7560, 8100):
        if i % 50 == 0:
            print(f'{i} / 540')
        pred = iio.imread(os.path.join(data_path, f'{str(i).zfill(6)}.mp4'), index=None)
        gt = iio.imread(
            os.path.join(
                "./dataset/video_annos/clips", 
                f'{str(i).zfill(6)}.mp4'
            ), 
            index=None
        )

        gt_list.append(gt)
        pred_list.append(pred[:33])
    print('test results loaded.')

    gt_list = np.stack(gt_list)
    pred_list = np.stack(pred_list)

    print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

    # image classification scores
    n_way = [2, 50]
    num_trials = 100
    top_k = 1
    # video classification scores
    acc_list, std_list = video_classify_metric(
                                        pred_list,
                                        gt_list,
                                        n_way = n_way,
                                        top_k=top_k,
                                        num_trials=num_trials,
                                        num_frames=gt_list.shape[1],
                                        return_std=True,
                                        device=device
                                        )
    for i, nway in enumerate(n_way):
        print(f'video classification score ({nway}-way): {np.mean(acc_list[i])} +- {np.mean(std_list[i])}')

    acc_aver = [[] for i in range(len(n_way))]
    acc_std  = [[] for i in range(len(n_way))]
    ssim_aver = []
    ssim_std  = []
    for i in range(pred_list.shape[1]):

        # ssim scores
        ssim_scores, std = ssim_score_only(pred_list[:, i], gt_list[:, i])
        ssim_aver.append(ssim_scores)
        ssim_std.append(std)

        print(f'ssim score: {ssim_scores}, std: {std}')
        
        acc_list, std_list = img_classify_metric(
                                            pred_list[:, i], 
                                            gt_list[:, i], 
                                            n_way = n_way, 
                                            top_k=top_k, 
                                            num_trials=num_trials, 
                                            return_std=True,
                                            device=device)
        for idx, nway in enumerate(n_way):
            acc_aver[idx].append(np.mean(acc_list[idx]))
            acc_std[idx].append(np.mean(std_list[idx]))
            print(f'img classification score ({nway}-way): {np.mean(acc_list[idx])} +- {np.mean(std_list[idx])}')
    
    print('----------------- average results -----------------')
    print(f'average ssim score: {np.mean(ssim_aver)}, std: {np.mean(ssim_std)}')
    for i, nway in enumerate(n_way):
        print(f'average img classification score ({nway}-way): {np.mean(acc_aver[i])} +- {np.mean(acc_std[i])}')

if __name__ == '__main__':
    main(
        data_path = 'brain_5b_sub05'
    )