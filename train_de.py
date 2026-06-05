import os
import psutil
import argparse
import torch.nn
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from dataloader import HideImage
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from models.networks import LatentDifferenceExpansion
from torch.utils.data import DataLoader, RandomSampler
from models.utils import PenaltyLoss, compute_psnr, overflow_num, quantize_residual_image, find_latest_model, GaussianPrior, LaplacePrior

torch.autograd.set_detect_anomaly(True)


def train(args):
    torch.manual_seed(args.seed)
    # logs
    log_path = os.path.join(args.logs_path, args.train_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    # create model
    model = LatentDifferenceExpansion(n_channels=args.c,
                                      base_filters=args.base_filters,
                                      use_cbam=args.use_cbam,
                                      clamp=args.clamp,
                                      scale=args.scale,
                                      bpp=args.bpp,
                                      k=args.k,
                                      block_mode=args.block_mode).to(args.device)
    # load model
    global_epoch = 0
    if args.continue_train:
        model_path = find_latest_model(f"{args.checkpoint_path}/", args.train_name)
        self, saved_result = model.load(model_path)
        start_epoch = saved_result['epoch']
        global_epoch = saved_result['global_epoch']
        args.lambda_penalty = saved_result['lambda_penalty']
        args.penalty_start_epoch = saved_result['penalty_start_epoch']
        args.interval_epoch = saved_result['interval_epoch']
        args.increase_penalty = saved_result['increase_penalty']
        args.interval_epoch = 25
        args_dict = vars(args)
        
        table = PrettyTable(["Argument", "Value"])
        table.add_row(["Training Mode", f"Continue Training (Resumed from {model_path})"])
        for arg, value in args_dict.items():
            table.add_row([arg, value])
        print(table)
    else:
        start_epoch = 0
        args_dict = vars(args)
        table = PrettyTable(["Argument", "Value"])
        table.add_row(["Training Mode", "Fresh Start (New Training)"])
        for arg, value in args_dict.items():
            table.add_row([arg, value])
        print(table)

    # datasets
    cpu_count = psutil.cpu_count(logical=True)
    num_workers = min(8, cpu_count - 1) if cpu_count > 4 else cpu_count
    dataset = HideImage(args.dataset_path, args.im_size)
    sampler = RandomSampler(dataset, replacement=False)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory_device='cuda'
    )

    # losses
    LaplacePriorLoss = LaplacePrior(target_loc=0, target_scale=1.)
    GaussianPriorLoss = GaussianPrior(target_mean=0, target_std=1.)
    MSELoss = torch.nn.MSELoss()
    PLoss = PenaltyLoss(max_value=255., penalty_loss_type=args.penalty_loss_type)
    penalty_increase_num_threshold = args.penalty_increase_threshold * args.im_size * args.im_size * args.c
    # optimizer
    optim_blocks = AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5)
    model.train()
    with tqdm(total=(args.num_epoch - start_epoch + 1), position=0, desc="Epoch", ncols=100, leave=True) as epoch_bar:
        under_overflow_avg_num = 1e9
        for epoch in range(args.num_epoch - start_epoch + 1):
            result = {}
            if epoch + start_epoch > args.penalty_start_epoch:
                if (epoch + start_epoch) % args.interval_epoch == 0 and under_overflow_avg_num > penalty_increase_num_threshold:
                    args.lambda_penalty += args.increase_penalty
            under_overflow_num_list = []
            for cover, secret in tqdm(train_loader, position=1, desc=f"Epoch {epoch + 1} Iter", ncols=100, leave=False):
                cover = cover.to(args.device)
                secret = secret.to(args.device)
                stego, _, z = model(cover, secret, args.hard_round, False)
                mse_loss = MSELoss(stego, cover)
                penalty_loss = PLoss(stego)
                if args.prior == "Gaussian":
                    z_loss = GaussianPriorLoss(z)
                elif args.prior == "Laplace":
                    z_loss = LaplacePriorLoss(z)
                else:
                    z_loss = 0.0
                total_loss = args.lambda_stego * mse_loss + args.lambda_z * z_loss + args.lambda_penalty * penalty_loss
                psnr = compute_psnr(stego, cover, max_value=255.)
                overflow_num_ave = overflow_num(stego, mode=255, min_value=0, max_value=255)
                underflow_num_avg = overflow_num(stego, mode=0, min_value=0, max_value=255)
                under_overflow_num_list.append(overflow_num_ave + underflow_num_avg)
                under_overflow_avg_num = np.mean(under_overflow_num_list)
                result["loss"] = {
                    "img_loss": mse_loss,
                    "z_loss": z_loss,
                    "penalty_loss": penalty_loss,
                    "total_loss": total_loss,
                    "psnr": psnr,
                    "lambda_penalty": args.lambda_penalty,
                    "under_overflow_num": under_overflow_avg_num,
                }
                result["output"] = {
                    "cover": cover,
                    "stego": stego,
                    "z": z,
                    "residual": quantize_residual_image(stego, cover, max_value=255.) * 255.
                }
                optim_blocks.zero_grad()
                total_loss.backward()
                optim_blocks.step()
                logs_train_loss_save(writer, result=result, global_step=global_epoch + 1)
                epoch_bar.set_description(desc=f"Epoch: {epoch + 1} / PSNR: {psnr:.2f}, Penalty: {args.lambda_penalty:.2f}, Pixels: {under_overflow_avg_num:.2f}, Total Loss: {total_loss:.4f}, Img Loss: {mse_loss:.4f}, Penalty Loss: {penalty_loss:.4f}")
                global_epoch += 1

            if epoch % args.save_epoch == 0:
                model.save(args, f"{args.checkpoint_path}/{args.train_name}", epoch + start_epoch + 1, global_epoch)
            logs_train_output_save(writer, result=result, global_step=epoch + start_epoch + 1)
            epoch_bar.update(1)


def logs_train_loss_save(writer, result=None, global_step=1):
    for key in result.keys():
        if key == "loss":
            for key_loss in result[key].keys():
                writer.add_scalar(f'{key}/{key_loss}', result[key][key_loss], global_step)


def logs_train_output_save(writer, result=None, global_step=1):
    """

    :param result:
    :param writer:
    :param global_step:
    :return:
    """
    for key in result.keys():
        if key == "output":
            b = result[key][list(result[key].keys())[0]].shape[0]
            index = np.random.randint(0, b)
            for key_output in result[key].keys():
                writer.add_image(f'{key}/{key_output}', result[key][key_output][index, :, :, :] / 255., global_step)

            # Add histogram for z based on channel count
            if 'z' in result[key]:
                z_tensor = result[key]['z'][index]  # Shape: (C, H, W)
                num_channels = z_tensor.shape[0]

                # Determine histogram tag based on number of channels
                if num_channels == 3:
                    hist_tag = f'{key}/z_histogram'  # RGB combined
                elif num_channels == 1:
                    hist_tag = f'{key}/z_histogram'  # Grayscale
                else:
                    hist_tag = f'{key}/z_histogram'

                # Add histogram (PyTorch will flatten all dimensions)
                writer.add_histogram(hist_tag, z_tensor, global_step)


def generate_train_name_verbose(args):
    params = {
        'name': args.train_name,
        'bpp': args.bpp,
        'prior': args.prior,
        'ls': args.lambda_stego,
        'lz': args.lambda_z,
        'lp': args.lambda_penalty,
        'plt': args.penalty_loss_type,
        'pse': args.penalty_start_epoch,
        'ie': args.interval_epoch,
        'ip': args.increase_penalty,
        'bm': args.block_mode,
        'cbam': args.use_cbam,
        'bf': args.base_filters,
        'k': args.k,
        'scale': args.scale,
        'clamp': args.clamp,
        'lr': args.lr
    }
    name_parts = [f"{k}{v}" for k, v in params.items()]
    train_name = "_".join(name_parts)
    return train_name


def train_print():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument('--dataset_path', type=str, default=r'/data/chenjiale/datasets/DIV2K/train')
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--hard_round', type=bool, default=False)
    parser.add_argument("--train_name", type=str, default="test")
    parser.add_argument('--save_epoch', type=int, default=50)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--c', type=int, default=3)
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--clamp', type=float, default=4.)
    parser.add_argument('--bpp', type=int, default=1)
    parser.add_argument('--use_cbam', type=bool, default=True)
    parser.add_argument('--block_mode', type=str, default="ResBlock")
    parser.add_argument('--base_filters', type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument('--num_epoch', type=int, default=20000)
    parser.add_argument('--prior', type=str, default="Laplace")
    parser.add_argument('--penalty_loss_type', type=str, default="MAE")
    parser.add_argument('--penalty_start_epoch', type=int, default=3000)
    parser.add_argument('--penalty_increase_threshold', type=float, default=5e-4)
    parser.add_argument('--interval_epoch', type=int, default=25)
    parser.add_argument('--increase_penalty', type=float, default=1.)
    parser.add_argument('--lambda_z', type=float, default=0)
    parser.add_argument('--lambda_stego', type=float, default=1.)
    parser.add_argument('--lambda_penalty', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--logs_path', type=str, default=r"logs")
    parser.add_argument('--continue_train', type=bool, default=False)
    args = parser.parse_args()
    # Set the device based on gpu_id
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")
    args.train_name = "LDE_" + generate_train_name_verbose(args)
    train(args)


if __name__ == "__main__":
    train_print()
