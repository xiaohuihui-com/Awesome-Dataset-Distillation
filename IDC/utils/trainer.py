import os
import numpy as np
import torch
import torch.utils.data
from .augment import DiffAug
from .visual import Plotter
from tqdm import tqdm


def random_indices(y, nclass=10, intraclass=False):
    n = len(y)
    if intraclass:
        index = torch.arange(n)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n)  # 返回0~n-1的数组，随机打乱
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def revise_module(weight_dict):
    """多GPU训练时载入权重，各module层中会多module.前缀，原因是调用了model = torch.nn.DataParallel(model).to(device)
    解决方案：修改权重文件，在各层加上module.前缀
    """
    new_weight_dict = {}
    for k, v in weight_dict.items():
        if not k.startswith("module."):  # 增加 module
            k = 'module.' + k
            new_weight_dict[k] = v
        else:  # 去掉 module
            k = k[7:]
            new_weight_dict[k] = v
    return new_weight_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def trainer(args, model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, logger=None,
            plotter=None):
    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    if args.pretrained:
        pretrained = os.path.join(args.save_dir, args.dataset, args.model, 'checkpoint.pth.tar')
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)

    if args.dsa:
        aug = DiffAug(strategy=args.dsa_strategy, batch=False)
        logger.info(f"Start training with DSA and {args.mixup} mixup")
    else:
        aug = None
        logger.info(f"Start training with base augmentation and {args.mixup} mixup")

    for epoch in range(cur_epoch + 1, args.epochs + 1):
        is_best = False
        acc1_tr, acc5_tr, loss_tr = train_epoch(args,
                                                model,
                                                train_loader,
                                                nclass,
                                                criterion,
                                                optimizer,
                                                device,
                                                epoch,
                                                aug,
                                                mixup=args.mixup)

        acc1, acc5, loss_val = validate(args, model, val_loader, criterion, epoch, device)

        logger.info(
            'epoch[{}/{}] train: Top1 {:.2f}  Top5 {:.2f}  Loss {:.3f} | test: Top1 {:.2f}  Top5 {:.2f}  Loss {:.3f}'
                .format(epoch, args.epochs, acc1_tr, acc5_tr, loss_tr, acc1, acc5, loss_val))
        if plotter != None:
            plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_acc5 = acc5
            # logger.info(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        if args.save_ckpt and (is_best or (epoch % 10 == 0)):
            state = {
                'epoch': epoch,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(args.save_dir, args.dataset, args.model, state, is_best)
        scheduler.step()

    return best_acc1, acc1


def train_epoch(args,
                model,
                train_loader,
                nclass,
                criterion,
                optimizer,
                device,
                epoch=0,
                aug=None,
                mixup='vanilla'):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    total = 0
    train_bar = tqdm(train_loader)
    for i, (input, target) in enumerate(train_bar):
        input = input.to(device)
        target = target.to(device)
        if aug != None:
            with torch.no_grad():
                input = aug(input)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        else:
            output = model(input)
            loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += len(input)
        train_bar.desc = "train epoch[{0}/{1}] ({2}/{3}) Loss: {loss.avg:.3f} Top1-acc: {top1.avg:.3f} Top5-acc: {top5.avg:.4f}" \
            .format(epoch, args.epochs, total, len(train_loader.dataset), loss=losses, top1=top1, top5=top5)
    return top1.avg, top5.avg, losses.avg


def validate(args, model, val_loader, criterion, epoch, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    total = 0
    val_bar = tqdm(val_loader)
    for i, (input, target) in enumerate(val_bar):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))
        total += len(input)
        val_bar.desc = "test  epoch[{0}/{1}] ({2}/{3}) Loss: {loss.avg:.3f} Top1-acc: {top1.avg:.3f} Top5-acc: {top5.avg:.4f}" \
            .format(epoch, args.epochs, total, len(val_loader.dataset), loss=losses, top1=top1, top5=top5)
    return top1.avg, top5.avg, losses.avg


def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        checkpoint['state_dict'] = dict(
            (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'(epoch: {}, best acc1: {}%)".format(
            path, cur_epoch, checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
        cur_epoch = 0
        best_acc1 = 100

    return cur_epoch, best_acc1


def save_checkpoint(save_dir, dataset, net, state, is_best):
    os.makedirs(os.path.join(save_dir, dataset, net), exist_ok=True)
    if is_best:
        ckpt_path = os.path.join(save_dir, dataset, net, 'model_best.pth.tar')
    else:
        ckpt_path = os.path.join(save_dir, dataset, net, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    print("checkpoint saved! ", ckpt_path)


class Trainer(object):
    """args:
            dataset,model,epochs
            save_dir,pretrained,save_ckpt
            mixup,mix_p,beta
            dsa,dsa_strategy
            plotter
        """

    def __init__(self, model, train_loader, test_loader, nclass, criterion, optimizer, scheduler, device,
                 args, aug=None):
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.nclass = nclass
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = os.path.join(args.save_dir, args.dataset, args.model)
        self.device = device
        self.args = args
        self.aug = aug
        self.plotter = Plotter(os.path.join(args.save_dir, args.dataset, args.model), args.epochs)

    def fit(self, logger):
        # Load pretrained
        cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
        if self.args.pretrained:
            path = os.path.join(self.save_dir, 'checkpoint.pth.tar')
            cur_epoch, best_acc1 = self.load_checkpoint(path)

        if self.args.dsa:
            self.aug = DiffAug(strategy=self.args.dsa_strategy, batch=False)
            logger.info(f"Start training with DSA and {self.args.mixup} mixup")
        else:
            self.aug = None
            logger.info(f"Start training with base augmentation and {self.args.mixup} mixup")

        for epoch in range(cur_epoch + 1, self.args.epochs + 1):
            is_best = False
            acc1_tr, acc5_tr, loss_tr = self.train(epoch)
            acc1, acc5, loss_val = self.validate(epoch)
            logger.info(
                'epoch[{}/{}] train: Top1 {:.2f}  Top5 {:.2f}  Loss {:.3f} | test: Top1 {:.2f}  Top5 {:.2f}  Loss {:.3f}'
                    .format(epoch, self.args.epochs, acc1_tr, acc5_tr, loss_tr, acc1, acc5, loss_val))
            if self.args.plotter:
                self.plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                # logger.info(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

            if self.args.save_ckpt and (is_best or (epoch % 10 == 0)):
                state = {
                    'epoch': epoch,
                    'arch': self.args.model,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'optimizer': self.optimizer.state_dict(),
                }
                self.save_checkpoint(state, is_best)
            self.scheduler.step()

        return best_acc1, acc1

    def train(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.train()
        total = 0
        train_bar = tqdm(self.train_loader)
        for i, (input, target) in enumerate(train_bar):
            input = input.to(self.device)
            target = target.to(self.device)
            if self.aug != None:
                with torch.no_grad():
                    input = self.aug(input)

            r = np.random.rand(1)
            if r < self.args.mix_p and self.args.mixup == 'cut':
                # generate mixed sample
                lam = np.random.beta(self.args.beta, self.args.beta)
                rand_index = random_indices(target, nclass=self.nclass)

                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                output = self.model(input)
                loss = self.criterion(output, target) * ratio + self.criterion(output, target_b) * (1. - ratio)
            else:
                output = self.model(input)
                loss = self.criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += len(input)
            train_bar.desc = "train epoch[{0}/{1}] ({2}/{3}) Loss: {loss.avg:.3f} Top1-acc: {top1.avg:.3f} Top5-acc: {top5.avg:.4f}" \
                .format(epoch, self.args.epochs, total, len(self.train_loader.dataset), loss=losses, top1=top1,
                        top5=top5)
        return top1.avg, top5.avg, losses.avg

    def validate(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        total = 0
        val_bar = tqdm(self.test_loader)
        for i, (input, target) in enumerate(val_bar):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.model(input)
            loss = self.criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            total += len(input)
            val_bar.desc = "test  epoch[{0}/{1}] ({2}/{3}) Loss: {loss.avg:.3f} Top1-acc: {top1.avg:.3f} Top5-acc: {top5.avg:.4f}" \
                .format(epoch, self.args.epochs, total, len(self.test_loader.dataset), loss=losses, top1=top1,
                        top5=top5)
        return top1.avg, top5.avg, losses.avg

    def load_checkpoint(self, path):
        #     try:
        #         load_weights_dict = {k: v for k, v in weights_dict.items()
        #                              if model.state_dict()[k].numel() == v.numel()}
        #     except:
        #         weights_dict = revise_module(weights_dict)
        #         load_weights_dict = {k: v for k, v in weights_dict.items()
        #                              if model.state_dict()[k].numel() == v.numel()}
        #     print(model.load_state_dict(load_weights_dict, strict=False))
        # else:
        #     raise FileNotFoundError("not found weights file: {}".format(opt.weights))
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            try:
                checkpoint['state_dict'] = dict(
                    (key, value) for (key, value) in checkpoint['state_dict'].items())
            except:
                checkpoint['state_dict'] = revise_module(checkpoint['state_dict'])
            print(self.model.load_state_dict(checkpoint['state_dict'], strict=False))
            self.model.load_state_dict(checkpoint['state_dict'])
            cur_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'(epoch: {}, best acc1: {}%)".format(
                path, cur_epoch, checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            cur_epoch = 0
            best_acc1 = 100

        return cur_epoch, best_acc1

    def save_checkpoint(self, state, is_best):
        os.makedirs(os.path.join(self.save_dir), exist_ok=True)
        if is_best:
            ckpt_path = os.path.join(self.save_dir, 'model_best.pth.tar')
        else:
            ckpt_path = os.path.join(self.save_dir, 'checkpoint.pth.tar')
        torch.save(state, ckpt_path)
        print("checkpoint saved! ", ckpt_path)


class Tester(object):
    """args:
            dataset,model,epochs
            save_dir,pretrained,save_ckpt
            mixup,mix_p,beta
            dsa,dsa_strategy
            plotter
        """

    def __init__(self, model, train_loader, test_loader, nclass, criterion, optimizer, scheduler, device,
                 args, aug=None):
        super(Tester, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.nclass = nclass
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = os.path.join(args.save_dir, args.dataset, args.model)
        self.device = device
        self.args = args
        self.aug = aug

    def fit(self, logger):
        # Load pretrained
        cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
        if self.args.pretrained:
            path = os.path.join(self.save_dir, 'checkpoint.pth.tar')
            cur_epoch, best_acc1 = self.load_checkpoint(path)

        if self.args.dsa:
            self.aug = DiffAug(strategy=self.args.dsa_strategy, batch=False)
            logger.info(f"Start training with DSA and {self.args.mixup} mixup")
        else:
            self.aug = None
            logger.info(f"Start training with base augmentation and {self.args.mixup} mixup")

        for epoch in range(cur_epoch + 1, self.args.epochs + 1):
            is_best = False
            acc1_tr, acc5_tr, loss_tr = self.train(epoch)
            if epoch % ((self.args.epochs + 1) // 4) == 0:
                acc1, acc5, loss_val = self.validate(epoch)
                logger.info(
                    'epoch[{}/{}] train: Top1 {:.2f}  Top5 {:.2f}  Loss {:.3f} | test: Top1 {:.2f}  Top5 {:.2f}  Loss {:.3f}'
                        .format(epoch, self.args.epochs, acc1_tr, acc5_tr, loss_tr, acc1, acc5, loss_val))
                is_best = acc1 > best_acc1
                if is_best:
                    best_acc1 = acc1
                    best_acc5 = acc5
            self.scheduler.step()
        return best_acc1, acc1

    def train(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.train()
        total = 0
        for i, (input, target) in enumerate(self.train_loader):
            input = input.to(self.device)
            target = target.to(self.device)
            if self.aug != None:
                with torch.no_grad():
                    input = self.aug(input)

            r = np.random.rand(1)
            if r < self.args.mix_p and self.args.mixup == 'cut':
                # generate mixed sample
                lam = np.random.beta(self.args.beta, self.args.beta)
                rand_index = random_indices(target, nclass=self.nclass)

                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                output = self.model(input)
                loss = self.criterion(output, target) * ratio + self.criterion(output, target_b) * (1. - ratio)
            else:
                output = self.model(input)
                loss = self.criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += len(input)

        return top1.avg, top5.avg, losses.avg

    def validate(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        total = 0
        for i, (input, target) in enumerate(self.test_loader):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.model(input)
            loss = self.criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            total += len(input)

        return top1.avg, top5.avg, losses.avg

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            try:
                checkpoint['state_dict'] = dict(
                    (key, value) for (key, value) in checkpoint['state_dict'].items())
            except:
                checkpoint['state_dict'] = revise_module(checkpoint['state_dict'])
            print(self.model.load_state_dict(checkpoint['state_dict'], strict=False))
            self.model.load_state_dict(checkpoint['state_dict'])
            cur_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'(epoch: {}, best acc1: {}%)".format(
                path, cur_epoch, checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            cur_epoch = 0
            best_acc1 = 100

        return cur_epoch, best_acc1

    def save_checkpoint(self, state, is_best):
        os.makedirs(os.path.join(self.save_dir), exist_ok=True)
        if is_best:
            ckpt_path = os.path.join(self.save_dir, 'model_best.pth.tar')
        else:
            ckpt_path = os.path.join(self.save_dir, 'checkpoint.pth.tar')
        torch.save(state, ckpt_path)
        print("checkpoint saved! ", ckpt_path)


def test_data(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, args, logger, repeat=3):
    best_acc_l = []
    acc_l = []
    for i in range(repeat):
        logger.info(f'repeat[{i + 1}/{repeat}]:')
        trainer = Tester(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, args)
        best_acc, acc = trainer.fit(logger)
        best_acc_l.append(best_acc)
        acc_l.append(acc)
    logger.info(
        f'Repeat {repeat} => Evalutation Best, last acc: {np.mean(best_acc_l):.2f} {np.mean(acc_l):.2f}\n')


def train_data(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, args, logger,
               repeat=1):
    best_acc_l = []
    acc_l = []
    for i in range(repeat):
        logger.info(f'repeat[{i + 1}/{repeat}]:')
        trainer = Trainer(model, train_loader, val_loader, nclass, criterion, optimizer, scheduler, device, args)
        best_acc, acc = trainer.fit(logger)
        best_acc_l.append(best_acc)
        acc_l.append(acc)
    logger.info(
        f'Repeat {repeat} => Best, last acc: {np.mean(best_acc_l):.2f} {np.mean(acc_l):.2f}\n')
