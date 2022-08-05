import os
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import get_model, get_optim, get_loss, get_scheduler
from utils.common import set_seed
from utils.dataloader import load_data
from utils.trainer import Trainer
from config import get_config, update_config
from config import settings
import logging.config

logging.config.dictConfig(settings.LOGGING_DIC)
logger = logging.getLogger()


def main(args, repeat=1):
    start_time = datetime.datetime.now()
    opt = get_config(args.config)
    update_config(opt, args.__dict__)
    logger.info('{}'.format(opt))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_ids
    set_seed(opt.seed)
    # get data
    train_loader, val_loader, nclass = load_data(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    logger.info("Using {} device.".format(device))
    logger.info('Using {} dataloader workers every process'.format(opt.num_workers))
    logger.info("Using {} {} images for training, {} images for validation." \
                .format(opt.dataset.upper(), len(train_loader.dataset), len(val_loader.dataset)))

    if opt.dataset == 'cifar100':
        opt.num_classes = 100
    elif opt.dataset in ['mnist','fashion']:
        opt.channel = 1
        opt.size = 28

    # get network
    net = opt.model
    model = get_model(net)(num_classes=opt.num_classes, channel=opt.channel, size=opt.size, **opt.model_params[net])
    logger.info('Creating model {}, model parameters: {:.2f}M' \
                .format(net, sum([p.data.nelement() for p in model.parameters()]) / 10 ** 6))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # get loss optimizer,schedule
    learn = opt.learning
    pg = [p for p in model.parameters() if p.requires_grad]
    criterion = get_loss(learn['loss'])().to(device)
    optimizer = get_optim(learn['optim'])(pg, **learn[learn['optim']])
    scheduler = get_scheduler(learn['scheduler'])(optimizer, milestones=[2 * opt.epochs // 3, 5 * opt.epochs // 6],
                                                  gamma=0.2)

    best_acc_l = []
    acc_l = []
    for i in range(repeat):
        logger.info(f"Repeat: {i + 1}/{repeat}")
        # plotter = Plotter(os.path.join(opt.save_dir, opt.dataset, opt.model), opt.epochs, idx=i)
        trainer = Trainer(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device,
                          opt)
        best_acc, acc = trainer.fit(logger)
        best_acc_l.append(best_acc)
        acc_l.append(acc)
    logger.info(f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_l):.2f} {np.mean(acc_l):.2f}')
    end_time = datetime.datetime.now()
    run_time = (end_time - start_time).total_seconds()
    logger.info('Using {} {} run time：{:.2f}h'.format(opt.dataset.upper(), opt.model, run_time / 3600.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--config", default='config.yaml')
    parser.add_argument('--dataset', type=str, \
                        default='cifar10',
                        choices=['mnist', 'fashion', 'svhn', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data_dir', type=str, default='../Efficient-Dataset-Condensation/data')
    parser.add_argument('--model', type=str)

    args = parser.parse_args()
    main(args)
