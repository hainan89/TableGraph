import os
import shutil
import torch
from collections import OrderedDict
import glob
import pandas as pd

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        train_cm_d = os.path.join(self.experiment_dir, 'train')
        val_cm_d = os.path.join(self.experiment_dir, 'val')
        test_cm_d = os.path.join(self.experiment_dir, 'test')
        if not os.path.exists(train_cm_d):
            os.makedirs(train_cm_d)

        if not os.path.exists(val_cm_d):
            os.makedirs(val_cm_d)

        if not os.path.exists(test_cm_d):
            os.makedirs(test_cm_d)


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')

        # save all configure args
        p = OrderedDict(vars(self.args))

        # config_dict = vars(self.args)
        # for k, v in config_dict.items():
        #     p[k] = config_dict[v]            
        # p['datset'] = self.args.dataset
        # p['backbone'] = self.args.backbone
        # p['out_stride'] = self.args.out_stride
        # p['lr'] = self.args.lr
        # p['lr_scheduler'] = self.args.lr_scheduler
        # p['loss_type'] = self.args.loss_type
        # p['epoch'] = self.args.epochs
        # p['base_size'] = self.args.base_size
        # p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

    def save_confusion_matrix(self, cm, epoch, cm_type):
        '''
        # cm_type: 'train', 'val', or 'test'
        '''
        cm_file = os.path.join(self.experiment_dir, cm_type, 'cm_{0}.csv'.format(epoch))
        pd.DataFrame(cm).to_csv(cm_file)
