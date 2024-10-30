from torch.cuda.amp import autocast as autocast
from seq_scripts import seq_inference
import utils
from utils import video_augmentation
from collections import OrderedDict
import inspect
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import faulthandler
import importlib
import torch
import yaml
import cv2
import os
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

faulthandler.enable()


class Processor():
    def __init__(self, arg):
        self.arg = arg

        if os.path.exists(self.arg.work_dir):
            answer = input(
                'Current dir exists, do you want to remove and refresh it?\n')
            if answer in ['yes', 'y', 'ok', '1']:
                print('Dir removed !')
                shutil.rmtree(self.arg.work_dir, ignore_errors=True)
                os.makedirs(self.arg.work_dir)
            else:
                print('Dir Not removed !')
        else:
            os.makedirs(self.arg.work_dir)

        if not self.arg.work_dir.endswith('/'):
            self.arg.work_dir = self.arg.work_dir + '/'

        shutil.copy2(__file__, self.arg.work_dir)
        shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
        copy_tree('slowfast_modules', self.arg.work_dir + 'slowfast_modules')
        copy_tree('modules', self.arg.work_dir + 'modules')

        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval)

        if self.arg.load_weights:
            self.load_slowfast_pkl = False
        else:
            self.load_slowfast_pkl = True

        self.device = utils.GpuDataParallel()
        self.gloss_dict = np.load(
            self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.arg.optimizer_args['num_epoch'] = self.arg.num_epoch
        slowfast_args = []
        for key, value in self.arg.slowfast_args.items():
            slowfast_args.append(key)
            slowfast_args.append(value)
        self.arg.slowfast_args = slowfast_args
        self.save_arg()
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.load_weights is None:
            print('Please appoint --weights.')
        self.recoder.print_log('Model:   {}.'.format(self.arg.model))
        self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
        vid, vid_lgt = self.load_data()
        seq_inference(vid, vid_lgt, self.model, self.device, self.recoder)
        self.recoder.print_log('Evaluation Done.\n')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
            load_pkl=self.load_slowfast_pkl,
            slowfast_config=self.arg.slowfast_config,
            slowfast_args=self.arg.slowfast_args
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        model = self.model_to_device(model)
        self.kernel_sizes = model.conv1d.kernel_size
        print("Loading model finished.")
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            raise ValueError(
                "AMP equipped with DataParallel has to manually write autocast() for each forward function, you can choose to do this by yourself")
            # model.conv2d = nn.DataParallel(model.conv2d, device_ids=self.device.gpu_list, output_device=self.device.output_device)
            # model = convert_model(model)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v)
                                 for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_data(self):
        print("Loading data")
        video_path = os.path.join(
            self.arg.dataset_info['dataset_root'], 'features/fullFrame-256x256px/inference/*.png')
        img_list = sorted(glob.glob(video_path))
        img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    for img_path in img_list]
        transform = video_augmentation.Compose([
            video_augmentation.CenterCrop(224),
            video_augmentation.Resize(1.0),
            video_augmentation.ToTensor(),
        ])
        vid, _ = transform(img_list, None, None)
        vid = ((vid.float() / 255.) - 0.45) / 0.225
        vid = vid.unsqueeze(0)

        left_pad = 0
        last_stride = 1
        total_stride = 1
        for _, ks in enumerate(self.kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride

        max_len = vid.size(1)
        video_length = torch.LongTensor(
            [np.ceil(vid.size(1) / total_stride).astype(np.int32) * total_stride + 2*left_pad])

        right_pad = int(np.ceil(max_len / total_stride)) * \
            total_stride - max_len + left_pad
        max_len = max_len + left_pad + right_pad
        vid = torch.cat(
            (
                vid[0, 0][None].expand(left_pad, -1, -1, -1),
                vid[0],
                vid[0, -1][None].expand(max_len -
                                        vid.size(1) - left_pad, -1, -1, -1),
            ), dim=0).unsqueeze(0)
        print("Loading data finished.")
        return vid, video_length


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    processor.start()
