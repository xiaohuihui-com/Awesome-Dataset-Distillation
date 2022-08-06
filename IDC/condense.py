import os
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from models import get_model, get_optim, get_loss, get_scheduler
from utils.common import set_seed, save_img
from utils.matchloss import matchloss
from utils.dataloader import load_resized_data, ClassMemDataLoader, ClassDataLoader
from utils.augment import diffaug
from utils.synth import Synthesizer
from utils.trainer import Trainer
from config import get_config, update_config
from config import settings
import logging.config
import warnings

warnings.filterwarnings("ignore")
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
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    trainset, val_loader = load_resized_data(opt)
    if opt.load_memory:
        train_loader = ClassMemDataLoader(trainset, batch_size=opt.batch_size, device=device)
    else:
        train_loader = ClassDataLoader(trainset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    logger.info("Using {} device.".format(device))
    logger.info('Using {} dataloader workers every process'.format(opt.num_workers))
    logger.info("Using {} {} images for training, {} images for validation." \
                .format(opt.dataset.upper(), len(train_loader.dataset), len(val_loader.dataset)))
    synset = Synthesizer(opt, nclass, nch, hs, ws, device)
    synset.init(train_loader, init_type=opt.init)
    save_img(os.path.join(opt.save_dir, 'init.png'),
             synset.data,
             unnormalize=False,
             dataname=opt.dataset)
    # Define augmentation function
    aug, aug_rand = diffaug(opt, device)
    save_img(os.path.join(opt.save_dir, f'aug.png'),
             aug(synset.sample(0, max_size=opt.batch_syn_max)[0]),
             unnormalize=True,
             dataname=opt.dataset)

    # evaluate
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
    # synset.test(opt,model, val_loader, nclass,
    #             criterion,
    #             optimizer,
    #             scheduler,
    #             device, logger, bench=False)
    optim_img = get_optim(learn['optim'])(synset.parameters(), lr=opt.lr_img, momentum=opt.mom_img)
    for it in range(opt.niter):
        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        for ot in range(opt.inner_loop):
            # Update synset
            for c in range(nclass):
                img, lab = train_loader.class_sample(c)
                img_syn, lab_syn = synset.sample(c, max_size=128)
                n = img.shape[0]
                img_aug = aug(torch.cat([img, img_syn]))
                loss = matchloss(opt, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                loss_total += loss.item()
                optim_img.zero_grad()
                loss.backward()
                optim_img.step()

            # Net update
            if opt.n_data > 0:
                for epoch in range(1):
                    trainer = Trainer(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler,
                                          device,opt)
                    trainer.train(epoch)
        if it % 10 == 0:
            logger.info(f"(Iter {it:3d}) loss: {loss_total / nclass / opt.inner_loop:.2f}")
        # if (it + 1) in it_test:
        #     save_img(os.path.join(args.save_dir, f'img{it + 1}.png'),
        #              synset.data,
        #              unnormalize=False,
        #              dataname=args.dataset)

    # end_time = datetime.datetime.now()
    # run_time = (end_time - start_time).total_seconds()
    # logger.info('Using {} {} run time：{:.3f}h'.format(opt.dataset.upper(), opt.model, run_time / 3600.0))


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
