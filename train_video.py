import argparse
import fcntl
import json
import os
import random
import sys
import time
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.datasets.video_dataset import FastVideoFolder
from src.models.video_model_train import DMC, VideoDataParallel
from src.models.image_model_train import DMCI
from src.utils.common import str2bool
from src.transforms.functional import ycbcr2rgb, rgb2ycbcr
from src.utils.stream_helper import get_state_dict

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script")

    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--train_patch_size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--train_frame_num', type=int, default=7)
    parser.add_argument('--train_crop_method', type=str, default='random')
    parser.add_argument('--train_frame_selection', type=str, default='random')
    parser.add_argument('--train_max_frame_distance', type=int, default=6)
    parser.add_argument('--train_random_flip', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--train_max_zoom_factor', type=float, default=1.0)
    parser.add_argument('--train_min_zoom_factor', type=float, default=1.0)

    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--test_patch_size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--test_frame_num', type=int, default=7)

    parser.add_argument('--i_frame_model_name', type=str, default="DMCI")
    parser.add_argument('--i_frame_model_path', type=str, required=True)
    parser.add_argument("--i_net_qp", type=int, default=0, help="Pretrained I net quality") # 0 21 42 63

    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('--training_scheduling', type=str, default='normal',
                        help='How to schedule the training strategy, support normal and fast')
    parser.add_argument('--lambda', dest='lmbda', type=float, default=-1)
    parser.add_argument('--num_epoch_per_checkpoint', type=int, default=20)

    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Dataloaders worker per trainer')
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--task_id', type=int, required=True, help='task id')
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='Use cuda')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save models')
    parser.add_argument('--seed', type=float, help='Set random seed for reproducibility')

    parser.add_argument("--Pyuv420", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("--Iyuv420", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("--load_checkpoint", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_checkpoint_setting', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='load pre-trained weights for faster training; ignored if there are '
                             'existing checkpoints')

    args = parser.parse_args(argv)
    return args


def get_loss_func(loss_func, lmbda):

    def loss_recon_mse(rd, fa_idx):
        return lmbda * rd['mse_loss']

    def loss_total_rdc_mse(rd, fa_idx):
        if fa_idx == 0:
            distortion_weight = 0.5
        elif fa_idx == 1:
            distortion_weight = 1.2
        elif fa_idx == 2:
            distortion_weight = 0.9
        return lmbda * distortion_weight * rd['mse_loss'] + rd['bpp']

    loss_func_name = f'loss_{loss_func}'
    assert loss_func_name in locals()
    return locals()[loss_func_name]


def get_training_strategy(training_scheduling):
    # epoch is for referencing purpose
    # lr is the learning rate for the current epoch
    # param could be inter, residue, both
    # loss could be me, me_mse, rec, rec_resibits, rec_bits
    # train_seq could be two, three, four, five, six, seven, nine, cascaded
    if training_scheduling == 'pretrain':
        training_strategy = \
            [[0,  1e-4,  "both",    "recon_mse",      "thirty_two"]]     * 6 +\
            [[6,  1e-4,  "both",    "recon_mse",      "three"]]   * 3 +\
            [[9,  1e-4,  "both",    "recon_mse",      "four"]]    * 3 +\
            [[12,  1e-4,  "both",    "recon_mse",      "six"]]    * 3 +\
            [[15,  1e-4,  "both",    "total_rdc_mse",  "two"]]     * 6 +\
            [[21,  1e-4,  "both",    "total_rdc_mse",  "three"]]   * 6 +\
            [[27,  1e-4,  "both",    "total_rdc_mse",  "four"]]    * 6 +\
            [[33,  1e-4,  "both",    "total_rdc_mse",  "six"]]    * 6 +\
            [[39,  1e-5,  "both",    "total_rdc_mse",  "six"]]     * 2
            # epo, lr,    param,     loss               train_seq
    elif training_scheduling == 'first_finetune':
        training_strategy = \
            [[0,   5e-5,  "both",    "total_rdc_mse",  "six_cascaded"]] * 4 +\
            [[4,   1e-5,  "both",    "total_rdc_mse",  "six_cascaded"]] * 6 +\
            [[10,  5e-5,  "both",    "total_rdc_mse",  "seven_cascaded"]] * 4 +\
            [[14,  1e-5,  "both",    "total_rdc_mse",  "seven_cascaded"]] * 6 +\
            [[20,  5e-6,  "both",    "total_rdc_mse",  "seven_cascaded"]] * 4 +\
            [[24,  1e-6,  "both",    "total_rdc_mse",  "seven_cascaded"]] * 4 +\
            [[28,  5e-7,  "both",    "total_rdc_mse",  "seven_cascaded"]] * 4 
            # epo, lr,    param,     loss               train_seq
    elif training_scheduling == 'second_finetune':
        training_strategy = \
            [[0,   1e-6,  "both",    "total_rdc_mse",  "nine_cascaded"]] * 80 +\
            [[80,  1e-6,  "both",    "total_rdc_mse",  "nine_cascaded"]] * 240 +\
            [[320, 1e-7,  "both",    "total_rdc_mse",  "nine_cascaded"]] * 180
            # epo, lr,    param,     loss               train_seq 
    # In slow or slower training presets,  users can finetune more epochs
    else:
        assert False

    return training_strategy


class LossRecorder():
    def __init__(self):
        super().__init__()
        self.loss = 0
        self.bpp_y_loss = 0
        self.bpp_z_loss = 0
        self.total_mse_loss = 0
        self.total_step = 0

    def update_loss(self, loss, info):
        self.loss += loss
        self.bpp_y_loss += info['bpp_y']
        self.bpp_z_loss += info['bpp_z']
        self.total_mse_loss += info['mse_loss']
        self.total_step += 1

    def print_average_loss(self, end='\n'):
        assert self.total_step > 0
        print(f'Loss: {self.loss/self.total_step:.4f} |'
              f'total_mse: {self.total_mse_loss/self.total_step:.6f} |'
              f'bpp_y: {self.bpp_y_loss/self.total_step:.6f} |'
              f'bpp_z: {self.bpp_z_loss/self.total_step:.6f} |', end=end)

    def add_to_tensor_board(self, writer, epoch, prefix='train'):
        assert self.total_step > 0
        writer.add_scalar(f"{prefix}_loss", self.loss / self.total_step, epoch)
        writer.add_scalar(f"{prefix}_bpp_y_loss", self.bpp_y_loss / self.total_step, epoch)
        writer.add_scalar(f"{prefix}_bpp_z_loss", self.bpp_z_loss / self.total_step, epoch)
        writer.add_scalar(f"{prefix}_total_mse_loss", self.total_mse_loss / self.total_step, epoch)


def train_one_epoch(video_net, i_frame_net, i_net_qp, task_id, lmbda, train_dataloader, optimizer, epoch,
                    writer, total_step, training_scheduling, rank, Pyuv420=False, Iyuv420=False):
    training_strategy = get_training_strategy(training_scheduling)
    i_frame_net.eval()
    video_net.train()
    device = next(video_net.parameters()).device

    idx = min(len(training_strategy)-1, epoch)
    lr = training_strategy[idx][1]

    for g in optimizer.param_groups:
        g['lr'] = lr

    loss_func = get_loss_func(training_strategy[idx][3], lmbda)

    train_seq = training_strategy[idx][4]
    loss_record = LossRecorder()

    if train_seq == 'two':
        train_dataloader.dataset.set_frame_num(2)
    elif train_seq=='thirty_two':
        train_dataloader.dataset.set_frame_num(32)
    elif train_seq == 'three':
        train_dataloader.dataset.set_frame_num(3)
    elif train_seq == 'four':
        train_dataloader.dataset.set_frame_num(4)
    elif train_seq == 'five':
        train_dataloader.dataset.set_frame_num(5)
    elif train_seq == 'six'or train_seq == 'six_cascaded':
        train_dataloader.dataset.set_frame_num(6)
    elif train_seq == 'seven' or train_seq == 'seven_cascaded':
        train_dataloader.dataset.set_frame_num(7)
    elif train_seq == 'nine' or train_seq == 'nine_cascaded':
        train_dataloader.dataset.set_frame_num(9)
    train_seq_length = train_dataloader.dataset.get_frame_num()

    train_cascaded_loss = False
    if train_seq == 'six_cascaded' or train_seq == 'seven_cascaded' or train_seq == 'nine_cascaded':
        train_cascaded_loss = True

    if train_cascaded_loss:
        video_net.set_noise_level(0.45)
    else:
        video_net.set_noise_level(0.4)

    # start_time = time.time()
    for i, d in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9, ncols=100):
        d = d.to(device)
        frame_nums = d.shape[1]//3

        ref_frame = d[:, 0:3, :, :]
        if train_seq_length > 3:
            with torch.no_grad():
                if Iyuv420 is False and Pyuv420 is True:
                    ref_frame = ycbcr2rgb(ref_frame)
                    result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                    ref_frame = result["x_hat"]
                    ref_frame = rgb2ycbcr(ref_frame)
                
                elif Iyuv420 is False and Pyuv420 is False:
                    ref_frame = rgb2ycbcr(ref_frame)
                    result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                    ref_frame = result["x_hat"]

                elif Iyuv420 is True and Pyuv420 is True:
                    result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                    ref_frame = result["x_hat"]
                else:
                    ref_frame = rgb2ycbcr(ref_frame)
                    result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                    ref_frame = result["x_hat"]
                    ref_frame = ycbcr2rgb(ref_frame)
        else:
            ref_frame = rgb2ycbcr(ref_frame)

        if train_cascaded_loss:
            optimizer.zero_grad(set_to_none=True)
            cur_frame = d[:, 3:frame_nums*3, :, :]
            result = video_net(cur_frame, ref_frame, None, 0, loss_func, Pyuv420)

            loss = result[0]
            info = result[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(video_net.parameters(), 1.0)
            optimizer.step()
        else:
            for frame_idx in range(1, frame_nums):
                cur_frame = d[:, frame_idx*3:(frame_idx+1)*3, :, :]
                optimizer.zero_grad(set_to_none=True)
                index_map = [0, 1, 0, 2]
                fa_idx = index_map[frame_idx % 4]
                result = video_net(cur_frame, ref_frame, None, fa_idx, loss_func, Pyuv420)
                ref_frame = None
                loss = result[0]
                info = result[1]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(video_net.parameters(), 1.0)
                optimizer.step()

        loss_record.update_loss(loss.item(), info)

        if i % 50 == 0 and rank <= 0:
            curr_loss = LossRecorder()
            curr_loss.update_loss(loss.item(), info)
            print(
                f"\ntask {task_id} rank {rank} total_step {total_step+loss_record.total_step} "
                f"train epoch {epoch}: ", end='')
            curr_loss.print_average_loss(end='')
            print(f"lr: {optimizer.param_groups[0]['lr']:.6f}")

        if rank >= 0 and i % 2000 == 0:
            # sync model
            ckpt_path = os.path.join("/dev", "shm", f"train_{task_id}_tmp", "model.checkpoint")
            if rank == 0:
                print(f"sync model with {ckpt_path}")
                torch.save(video_net.state_dict(), ckpt_path)
            dist.barrier(device_ids=[device.index])
            video_net.load_state_dict(torch.load(ckpt_path, map_location=device))
            dist.barrier(device_ids=[device.index])
            if rank == 0:
                os.remove(ckpt_path)
            dist.barrier(device_ids=[device.index])

    assert loss_record.total_step > 0

    if rank <= 0:
        loss_record.add_to_tensor_board(writer, epoch, 'train')
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

    return loss_record.total_step


def test_epoch(video_net, i_frame_net, i_net_qp, task_id, lmbda, test_dataloader, epoch, writer, rank, Pyuv420=False, Iyuv420=False):
    i_frame_net.eval()
    video_net.eval()
    device = next(video_net.parameters()).device

    loss_record = LossRecorder()
    start_time = time.time()
    with torch.no_grad():
        for _, d in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9, ncols=100):
            d = d.to(device)
            frame_nums = d.shape[1]//3

            ref_frame = d[:, 0:3, :, :]

            if Iyuv420 is False and Pyuv420 is True:
                ref_frame = ycbcr2rgb(ref_frame)
                result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                ref_frame = result["x_hat"]
                ref_frame = rgb2ycbcr(ref_frame)
            
            elif Iyuv420 is False and Pyuv420 is False:
                ref_frame = rgb2ycbcr(ref_frame)
                result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                ref_frame = result["x_hat"]

            elif Iyuv420 is True and Pyuv420 is True:
                result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                ref_frame = result["x_hat"]
            else:
                ref_frame = rgb2ycbcr(ref_frame)
                result = i_frame_net.get_rec_only(ref_frame, i_net_qp)
                ref_frame = result["x_hat"]
                ref_frame = ycbcr2rgb(ref_frame)

            ref_feature = None

            for frame_idx in range(1, frame_nums):
                cur_frame = d[:, frame_idx*3:(frame_idx+1)*3, :, :]
                index_map = [0, 1, 0, 2]
                fa_idx = index_map[frame_idx % 4]
                result = video_net(cur_frame, ref_frame, ref_feature, fa_idx, Pyuv420=Pyuv420)
                info = video_net.get_rd_info(result)
                loss = lmbda * info['mse_loss'].item() + info['bpp'].item()
                loss_record.update_loss(loss, info)

                ref_frame = result['recon_image'].detach()
                ref_feature = result["feature"].detach()

    if rank <= 0:
        print(f"\ntask {task_id} rank {rank} Test epoch {epoch}:"
              f"test time {(time.time()-start_time)}", end='')
        loss_record.print_average_loss()
        loss_record.add_to_tensor_board(writer, epoch, prefix='test')

    return loss_record.loss / loss_record.total_step


def get_latest_checkpoint_path(dir_cur):
    files = os.listdir(dir_cur)
    all_best_checkpoints = []
    for file in files:
        if file[-4:] == '.pth' and 'cur_' in file:
            all_best_checkpoints.append(os.path.join(dir_cur, file))
    if len(all_best_checkpoints) > 0:
        return max(all_best_checkpoints, key=os.path.getmtime)

    return 'not_exist'


def all_models_tested(result):
    for cls in result:
        for seq in result[cls]:
            if len(list(result[cls][seq].keys())) < 4:
                return False
    return True


def test_bd_rate(anchor_path, ckpt_result_path, ckpt_name, output_path):
    anchor_name = os.path.basename(anchor_path)[:-5]
    cmd = 'python compare_rd_video.py'
    cmd += ' --compare_between class'
    cmd += f' --base_method {anchor_name}'
    cmd += f' --output_path {output_path}'
    cmd += ' --plot_rd_curve 0'
    cmd += f' --log_paths {anchor_name} {anchor_path}'
    cmd += f' {ckpt_name} {ckpt_result_path}'
    cmd += ' --auto_test 1'
    os.system(cmd)

def main_video_net(argv):
    args = parse_args(argv)

    if args.cuda_idx is not None and len(args.cuda_idx) > 1:
        cuda_device = ','.join([str(s) for s in args.cuda_idx])
        print("task", args.task_id, "device", cuda_device)
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    if args.training_scheduling == 'normal':
        assert args.epochs > 40

    use_ddp = False
    if args.cuda and torch.cuda.device_count() > 0 and len(args.cuda_idx) > 1:
        use_ddp = True
    print("use_ddp:", use_ddp)

    if not use_ddp:
        train(-1, args)
    else:
        tmp_path = os.path.join("/dev", "shm", f"train_{args.task_id}_tmp")
        if not os.path.exists(tmp_path):
            print(f"create tmp folder: {tmp_path}")
            os.makedirs(tmp_path)
        else:
            print(f"{tmp_path} exists")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355 + args.task_id)
        world_size = len(args.cuda_idx)
        mp.spawn(train, nprocs=world_size, args=(args,), join=True)


def train(rank, args):
    torch.backends.cudnn.enabled = True
    if rank >= 0:
        world_size = len(args.cuda_idx)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
    train_dataset = FastVideoFolder(args.train_dataset,
                                    args.train_patch_size[0], args.train_patch_size[1],
                                    args.train_frame_num,
                                    crop_method=args.train_crop_method,
                                    frame_selection=args.train_frame_selection,
                                    max_frame_distance=args.train_max_frame_distance,
                                    max_zoom_factor=args.train_max_zoom_factor,
                                    min_zoom_factor=args.train_min_zoom_factor,
                                    random_flip=args.train_random_flip,
                                    Pyuv420=args.Pyuv420)
    test_dataset = FastVideoFolder(args.test_dataset,
                                   args.test_patch_size[0], args.test_patch_size[1],
                                   args.test_frame_num,
                                   disable_random=True,
                                   Pyuv420=args.Pyuv420)

    if rank >= 0:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
        assert args.train_batch_size % world_size == 0
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size // world_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            sampler=train_sampler,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    if rank <= 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    if rank >= 0:
        dist.barrier(device_ids=[rank])

    writer = None
    if rank <= 0:
        writer = SummaryWriter(f"{args.save_dir}")

    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_frame_net = DMCI()
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.eval()

    video_net = DMC()

    best_epoch = -1
    total_step = 0
    best_loss = 1e10
    begin_epoch = 0
    existing_checkpoint_path = get_latest_checkpoint_path(args.save_dir)
    if args.load_checkpoint and os.path.exists(existing_checkpoint_path):
        load_checkpoint = torch.load(existing_checkpoint_path, map_location=torch.device('cpu'))
        load_model_para = None
        try:
            load_model_para = load_checkpoint['state_dict']
            if args.use_checkpoint_setting:
                begin_epoch = load_checkpoint['epoch'] + 1
            # begin_epoch = 39
            total_step = load_checkpoint['total_step']
            best_loss = load_checkpoint['loss']
        except Exception:
            pass

        video_net.load_dict(load_model_para)
        print(f"matched result task {args.task_id}")
        print(f"****task {args.task_id} load model from {existing_checkpoint_path}"
              f" best_loss {best_loss} begin_epoch {begin_epoch} total_step {total_step} ****")
        if args.pretrained_weights:
            print('pretrained weights ignored due to existing checkpoints')
    elif args.pretrained_weights:
        print(f'loading pretrained weights from {args.pretrained_weights}')
        load_checkpoint = torch.load(args.pretrained_weights, map_location=torch.device('cpu'))
        if 'state_dict' in load_checkpoint:
            load_model_para = load_checkpoint['state_dict']
            # if args.use_checkpoint_setting:
            begin_epoch = load_checkpoint['epoch'] + 1
            # begin_epoch = 39
            total_step = load_checkpoint['total_step']
            best_loss = load_checkpoint['loss']
        video_net.load_dict(load_model_para, strict=False)

    print(f"cuda_is_available {torch.cuda.is_available()} device_count {torch.cuda.device_count()}")

    if rank >= 0:
        device = f"cuda:{rank}"
    elif args.cuda and torch.cuda.device_count() > 0:
        assert len(args.cuda_idx) == 1
        device = f"cuda:{args.cuda_idx[0]}"
    else:
        device = "cpu"
    print(f"device_type: {device}")

    if rank >= 0:
        video_net = video_net.cuda(rank)
        video_net = VideoDataParallel(video_net, device_ids=[rank],
                                      find_unused_parameters=True)
        i_frame_net = i_frame_net.cuda(rank)
        i_frame_net = VideoDataParallel(i_frame_net, device_ids=[rank],
                                        find_unused_parameters=True)
    else:
        video_net = video_net.to(device)
        i_frame_net = i_frame_net.to(device)

    optimizer = optim.AdamW(video_net.parameters(), lr=1e-4)

    for epoch in range(begin_epoch, args.epochs):
        if rank >= 0:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_step = train_one_epoch(
            video_net,
            i_frame_net,
            args.i_net_qp,
            args.task_id,
            args.lmbda,
            train_dataloader,
            optimizer,
            epoch,
            writer,
            total_step,
            args.training_scheduling,
            rank,
            Pyuv420=args.Pyuv420,
            Iyuv420=args.Iyuv420
        )

        total_step += epoch_step
        print(f"task {args.task_id} rank {rank} finish epoch {epoch}"
              f" epoch_step {epoch_step} total_step {total_step}")

        eopch_test_loss = test_epoch(
            video_net,
            i_frame_net,
            args.i_net_qp,
            args.task_id,
            args.lmbda,
            test_dataloader,
            epoch,
            writer,
            rank,
            Pyuv420=args.Pyuv420,
            Iyuv420=args.Iyuv420
        )

        if rank <= 0:  # rank < 0 means DDP is not used
            name_common = f"{args.save_dir.split('/')[-2]}_{args.save_dir.split('/')[-1]}"
            cur_checkpoint_name = f"{args.save_dir}/cur_{name_common}epo_{epoch}.pth"
            pre_checkpoint_name = f"{args.save_dir}/cur_{name_common}epo_{epoch-1}.pth"

            print(f"task {args.task_id} save model, eopch_test_loss {eopch_test_loss}")
            save_dict = {
                "epoch": epoch,
                "total_step": total_step,
                "state_dict": video_net.state_dict(),
                "loss": eopch_test_loss,
            }

            torch.save(save_dict, cur_checkpoint_name)
            if os.path.exists(pre_checkpoint_name):
                os.remove(pre_checkpoint_name)

            if epoch % args.num_epoch_per_checkpoint == 0 or epoch == args.epochs - 1:
                checkpoint_name = f"{args.save_dir}/checkpoint_{name_common}epo_{epoch}.pth"
                torch.save(save_dict, checkpoint_name)

            if eopch_test_loss < best_loss:
                best_ckpt_name = f"{args.save_dir}/best_{name_common}epo_{epoch}.pth"
                best_pre_ckpt_name = f"{args.save_dir}/best_{name_common}epo_{best_epoch}.pth"
                if os.path.exists(best_pre_ckpt_name):
                    os.remove(best_pre_ckpt_name)
                best_loss = eopch_test_loss
                torch.save(save_dict, best_ckpt_name)
                best_epoch = epoch

        if rank >= 0:
            dist.barrier(device_ids=[rank])

    if rank <= 0:
        writer.close()
    if rank >= 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    main_video_net(sys.argv[1:])
