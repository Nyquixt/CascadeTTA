# An implementation of MER Algorithm 1 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .common import *
from .utils import *

from torch.nn.modules.loss import CrossEntropyLoss
from metann import Learner
import os
import copy

class Net(nn.Module):
    def __init__(self, n_outputs, exp_name, args):
        super(Net, self).__init__()
        self.args = args
        self.exp_name = exp_name

        # define network
        if args.group_norm == 0: # default value
            norm_layer = nn.BatchNorm2d
        else:
            def gn_helper(planes):
                return nn.GroupNorm(args.group_norm, planes)
            norm_layer = gn_helper

        self.net = ResNetCifar(args.depth, args.width, channels=3, classes=n_outputs, norm_layer=norm_layer).cuda()
        
        if args.pretrained: # load meta-pretrained model
            self.load_model()
        elif args.erm_pretrained: # load erm-pretrained model
            self.load_erm_model()

        self.net = Learner(self.net) # wrap network into MetaNN Learner class
        # loss functions
        self.bce = CrossEntropyLoss()
        self.entropy = HLoss()

        param_feature = []
        param_bn = []
        param_main = [] # theta_m
        param_aux = [] # theta_s

        # 4 components in the network
        for name, param in self.net.named_parameters():
            if "bn" in name:
                param_bn.append(param)
            elif "fc" in name:
                param_main.append(param)
            elif "ssh" in name:
                param_aux.append(param)
            else:
                param_feature.append(param)

        if args.opt == 'sgd':
            # During adaptation
            self.opt_adaptation = torch.optim.SGD(param_bn + param_main,
                                        args.online_lr, momentum=args.momentum, 
                                        nesterov=args.nesterov,
                                        weight_decay=args.weight_decay)

            # Training optimizer
            self.opt_all = torch.optim.SGD(list(self.net.parameters()), 
                                        args.lr, momentum=args.momentum, 
                                        nesterov=args.nesterov,
                                        weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            # During adaptation
            self.opt_adaptation = torch.optim.Adam(param_bn + param_main,
                                        args.online_lr, weight_decay=args.weight_decay)

            # Training optimizer
            self.opt_all = torch.optim.Adam(list(self.net.parameters()), 
                                        args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception('optimizer not implemented')

        if self.args.pretrained: # just adapt
            self.source_model = copy.deepcopy(self.net)
            self.source_opt = copy.deepcopy(self.opt_adaptation)

        self.bsz = args.batch_size

        # self-supervised and meta loss weights
        self.ss = args.ss
        self.meta_w = args.meta_w

        self.n_meta = args.n_meta
        self.inner_steps = args.inner_steps # inner steps during pre-train
        self.grad_step_online = args.grad_step_online # online steps

        # handle gpus if specified
        if args.cuda:
            self.net = self.net.cuda()

    def inner_update(self, x, params, inner_lr): # simulate the adaptation process
        pred_H = self.net.functional(params.values(), True, x, self_super=True)[1] # from ss head
        loss_H = self.entropy(pred_H).mean(0)
        grads = torch.autograd.grad(loss_H, params.values(), create_graph=True, allow_unused=True)

        updated_params = copy.deepcopy(params)

        for key, grad in zip(params.keys(), grads):
            if "fc" in key or "bn" in key:
                # not updating ss head here
                updated_params[key] = (params.get(key) - inner_lr * grad).requires_grad_()
            else:
                updated_params[key] = (params.get(key)).requires_grad_()

        return updated_params

    def meta_loss(self, x, y, params):
        # main task and ss head
        pred, pred_H = self.net.functional(params, True, x, self_super=True)
        # update BN affine as well in CE loss
        meta_loss = self.bce(pred, y) + self.ss * self.entropy(pred_H).mean(0)
        return meta_loss

    def erm_loss(self, x, y, params):
        # old params first order gradients
        pred = self.net.functional(params, True, x)
        erm_loss = self.bce(pred, y)
        return erm_loss

    def pretrain(self, pretrain_loader, val_loader, logfile):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_all, self.args.milestones, gamma=0.1, last_epoch=-1)
        best_acc = -float('inf')
        for epoch in range(self.args.start_epoch, self.args.epochs):

            for param_group in self.opt_all.param_groups: # update lr
                lr = param_group['lr']
                inner_lr = self.args.inner_w * param_group['lr']
                
            print(f"inner lr: {inner_lr:.5f}; lr: {lr:.5f}")

            for i, (x, y) in enumerate(pretrain_loader):
                x = x.cuda()        
                y = y.cuda()
                
                for _ in range(self.n_meta):
                    
                    if epoch >= self.args.erm_epochs:
                        half = x.size(0) // 2
                        meta_train_x, meta_test_x = x[:half, :], x[-half:, :]
                        meta_train_y, meta_test_y = y[:half], y[-half:]
                        # meta-learn
                        old_params = list(self.net.parameters())
                        new_params = dict(self.net.named_parameters())
                        for _ in range(self.inner_steps):
                            new_params = self.inner_update(meta_train_x, new_params, inner_lr) # inner

                        meta_loss = self.meta_w * self.meta_loss(meta_test_x, meta_test_y, new_params.values()) + self.erm_loss(meta_train_x, meta_train_y, old_params) # outer
                        
                        # backprop
                        self.opt_all.zero_grad()
                        meta_loss.backward()
                        self.opt_all.step()
                    else:
                        # ERM
                        params = list(self.net.parameters())
                        pred = self.net.functional(params, True, x)
                        loss = self.bce(pred, y)

                        # backprop
                        self.opt_all.zero_grad()
                        loss.backward()
                        self.opt_all.step()

            scheduler.step()
            # evaluate on validation set per epoch
            prec1 = validate(val_loader, self.net, self.bce, epoch)
            info = str(epoch) + ' Validation set acc: ' + str(prec1)
            print(info)
            logfile.write(info + '\n')
            if prec1 > best_acc:
                print("Saving best network...\n")
                best_acc = prec1
                state = {'net': self.net.state_dict(), 'optimizer': self.opt_all.state_dict()}
                torch.save(state, self.args.outf + '/cifar10_' + self.exp_name + '.pth')

        # store source model and source optim
        self.source_model = copy.deepcopy(self.net)
        self.source_opt = copy.deepcopy(self.opt_adaptation)

        return meta_loss

    def adapt(self, x):
        
        self.net.train()
        for _ in range(self.n_meta):

            for i in range(self.grad_step_online):
                params = list(self.net.parameters())
                _, out_s = self.net.functional(params, True, x, self_super=True) # entropy from ss head

                loss = self.entropy(out_s).mean(0)

                # BN+\theta_m
                self.opt_adaptation.zero_grad()
                loss.backward()
                self.opt_adaptation.step()

        self.net.eval()
        params = list(self.net.parameters())
        out_m = self.net.functional(params, False, x)

        return out_m

    def reset(self):
        print("=> Resetting to source model...")
        self.net = copy.deepcopy(self.source_model)
        self.opt_adaptation = copy.deepcopy(self.source_opt)

    def configure_bn_params(self):
        self.net.train()
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def load_model(self):
        model_path = "./checkpoints/%s" % (self.args.pretrained)
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            state_dict = torch.load(model_path)['net']
            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                new_key = new_key.replace("stateless.", "")
                state_dict[new_key] = state_dict.pop(key)
            
            self.net.load_state_dict(state_dict)

    def load_erm_model(self):
        model_path = "./checkpoints/%s" % (self.args.erm_pretrained)
        if os.path.isfile(model_path):
            print("=> loading ERM checkpoint '{}'".format(model_path))

            state_dict = torch.load(model_path)['net']
            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)
            
            self.net.load_state_dict(state_dict, strict=False)
