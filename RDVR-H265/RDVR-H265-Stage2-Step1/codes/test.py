import os.path as osp
import logging
import time
import argparse
import csv
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

import time
import torch

def cal_ms_ssim(sr_img, gt_img, lr_img, lrgt_img,bpp_total2):
    # save images
    suffix = opt['suffix']
    if suffix:
        save_img_path = osp.join(dataset_dir, 'HR', folder, img_name + suffix + '.png')
    else:
        save_img_path = osp.join(dataset_dir, 'HR', folder, img_name + '.png')
    util.save_img(sr_img, save_img_path)
    #
    # if suffix:
    #     save_img_path = osp.join(dataset_dir, folder, img_name + suffix + '_GT.png')
    # else:
    #     save_img_path = osp.join(dataset_dir, folder, img_name + '_GT.png')
    # util.save_img(gt_img, save_img_path)
    #
    if suffix:
        ###save_img_path = osp.join(dataset_dir, folder, img_name + suffix + '_LR.png')
        save_img_path = osp.join(dataset_dir, 'LR', folder, img_name + suffix + '.png')
    else:
        ###save_img_path = osp.join(dataset_dir, folder, img_name + '_LR.png')
        save_img_path = osp.join(dataset_dir, 'LR', folder, img_name + '.png')

    util.save_img(lr_img, save_img_path)
    #
    # if suffix:
    #     save_img_path = osp.join(dataset_dir, folder, img_name + suffix + '_LR_ref.png')
    # else:
    #     save_img_path = osp.join(dataset_dir, folder, img_name + '_LR_ref.png')
    # util.save_img(lrgt_img, save_img_path)

    # calculate RGB MS-SSIM
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.

    lr_img = lr_img / 255.
    lrgt_img = lrgt_img / 255.

    crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
    if crop_border == 0:
        cropped_sr_img = sr_img
        cropped_gt_img = gt_img
    else:
        cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

    ms_ssim = util.calculate_ms_ssim(sr_img * 255/255, gt_img * 255/255)
    test_results['ms_ssim'].append(ms_ssim)

    # MS-SSIM for RGB LR
    ms_ssim_lr = util.calculate_ms_ssim(lr_img * 255, lrgt_img * 255)
    test_results['ms_ssim_lr'].append(ms_ssim_lr)

    test_results['bpp_total2']=test_results['bpp_total2']+bpp_total2
    logger.info('{:20s} - HR RGB MS_SSIM: {:.6f}. H.265 Decoded LR RGB MS-SSIM: {:.6f}.'.format(
         osp.join(folder, img_name), ms_ssim, ms_ssim_lr))

    return test_results

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    print(test_loader)
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    # util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['ms_ssim'] = []
    test_results['ms_ssim_lr'] = []
    
    test_results['bpp_total2']=0

    with open(osp.join(opt['path']['log'], 'test_' + opt['name'] + '_test.csv'), 'w') as f:
        writer = csv.writer(f)
        for data in test_loader:
            model.feed_data(data)
            if test_set_name == 'Vid4':
                folder = osp.split(osp.dirname(data['GT_path'][0][0]))[1]
            else:
                folder = ''
            util.mkdir(osp.join(dataset_dir, 'HR', folder))
            util.mkdir(osp.join(dataset_dir, 'LR', folder))

            t0 = time.time()

            model.test()
            t1 = time.time()
            print("===> Timer: %.4f sec." % (t1 - t0))  

            visuals = model.get_current_visuals()

            if test_set_name == 'Vimeo90K':
                center = visuals['SR'].shape[0] // 2
                img_path = data['GT_path'][0]
                img_name = osp.splitext(osp.basename(img_path))[0]

                sr_img = util.tensor2img(visuals['SR'])  # uint8
                gt_img = util.tensor2img(visuals['GT'][center])  # uint8
                lr_img = util.tensor2img(visuals['LR'])  # uint8
                lrgt_img = util.tensor2img(visuals['LR_ref'][center])  # uint8
                
                bpp_total2=visuals['bpp_total']
                
                test_results = cal_ms_ssim(sr_img, gt_img, lr_img, lrgt_img,bpp_total2)

            else:
                t_step = visuals['SR'].shape[0]
                for i in range(t_step):
                    img_path = data['GT_path'][i][0]
                    img_name = osp.splitext(osp.basename(img_path))[0]

                    sr_img = util.tensor2img(visuals['SR'][i])  # uint8
                    gt_img = util.tensor2img(visuals['GT'][i])  # uint8
                    lr_img = util.tensor2img(visuals['LR'][i])  # uint8
                    lrgt_img = util.tensor2img(visuals['LR_ref'][i])  # uint8

                    bpp_total2=visuals['bpp_total']                

                    test_results = cal_ms_ssim(sr_img, gt_img, lr_img, lrgt_img,bpp_total2)
                    
                    #logger.info('bpp_total: {:.6f}.'.format(bpp_total2))

    # Average MS-SSIM results
    ave_ms_ssim = sum(test_results['ms_ssim']) / len(test_results['ms_ssim'])
    ave_ms_ssim_lr = sum(test_results['ms_ssim_lr']) / len(test_results['ms_ssim_lr'])
    
    average_bpp=test_results['bpp_total2']/len(test_results['ms_ssim_lr'])
    
    logger.info(
        '----Average HR RGB MS-SSIM results for {}----\n\tMS-SSIM: {:.6f}. H.265 Decoded LR RGB MS-SSIM: {:.6f}; bpp: {:.6f}.\n'.format(
           test_set_name, ave_ms_ssim, ave_ms_ssim_lr, average_bpp))