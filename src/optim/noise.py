from contextlib import nullcontext
import math

import torch
import torch.distributed as dist
import wandb
import numpy as np
import random

from .utils import load_checkpoint, grad_shape, sample_gradient, gather_alpha, alpha_estimator, \
    nuclear_norm_of_positive_diagonal_matrix, produce_v, single_step, compute_heavy_tail_norm


def eval_noise(model, opt, datareaders, scheduler, cfg, exp_dir, distributed_backend, noise_reader, save_cnt, sample_idx_num = 25, sample_itr = 50):
    if cfg.sample_itr:
        sample_itr = cfg.sample_itr
    train_reader, val_reader = datareaders["train"], datareaders["val"]

    log_dict : dict= {}

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    noise_analysis_steps = np.linspace(0, cfg.iterations, num=save_cnt, dtype=int).tolist()
    # noise_analysis_steps = [0]
    sample_indexes = None

    for step in noise_analysis_steps:
        ckpt_dir = exp_dir / "ckpts" / f"itr_{step}"
        if not ckpt_dir.exists():
            print("no checkpoint found at {}".format(ckpt_dir))
            raise ValueError

        load_checkpoint(model, opt, scheduler, ckpt_dir / "main.pt", cfg.device)

        model.eval()
        device = cfg.device
        is_master : bool = distributed_backend.is_master_process()

        ori_shape = grad_shape(cfg, model, noise_reader, device, type_ctx, distributed_backend)
        M = ori_shape[0] * ori_shape[1]

        noise_at_ckpt , nabla_at_ckpt = sample_gradient(cfg = cfg,
                                       M = M,
                                       device = device,
                                       noise_reader = noise_reader,
                                       model = model ,
                                       distributed_backend = distributed_backend ,
                                       type_ctx = type_ctx,
                                       sample_per_rank = 1250)

        local_alpha_at_ckpt : torch.Tensor = alpha_estimator(noise_at_ckpt)
        print(f"local alpha of rank {dist.get_rank()} at iteration {step} : {local_alpha_at_ckpt}")
        avg_alpha_at_ckpt : float = gather_alpha(local_alpha_at_ckpt, distributed_backend, device)

        step_name = f"{step}"

        noise_norm : torch.Tensor= noise_at_ckpt.norm(p=avg_alpha_at_ckpt, dim=1).to(device)

        coordinate_norms : dict = {}
        coordinate_alphas : dict = {}
        N, d = noise_at_ckpt.shape

        if sample_indexes is None:
            random.seed(42)
            sample_indexes = random.sample(range(0, d), sample_idx_num)
            # print("sample_indexes:", sample_indexes)

        for sample_idx in sample_indexes:
            coordinate_noise_at_ckpt = noise_at_ckpt[:, sample_idx].reshape(-1, 1)
            coordinate_alpha_at_ckpt = alpha_estimator(coordinate_noise_at_ckpt)
            avg_coordinate_alpha_at_ckpt = gather_alpha(coordinate_alpha_at_ckpt, distributed_backend, device)
            coordinate_alphas[sample_idx] = avg_coordinate_alpha_at_ckpt

            if avg_coordinate_alpha_at_ckpt < 1.1:
                coordinate_norms[sample_idx] = torch.abs(coordinate_noise_at_ckpt) ** avg_coordinate_alpha_at_ckpt
                if is_master:
                    print(f"{sample_idx} is a heavy-tailed index, with p = {avg_coordinate_alpha_at_ckpt}")

            if is_master:
                print(f"coordinate_alpha at coord {sample_idx}: {avg_coordinate_alpha_at_ckpt}")


        m, n = ori_shape
        nuclear_norm_of_v = 0
        noise_in_matrix = noise_at_ckpt.reshape(N, m, n).detach().cpu()
        v = produce_v(noise_in_matrix, avg_alpha_at_ckpt)
        nuclear_norm_of_v = nuclear_norm_of_positive_diagonal_matrix(v)
        del noise_in_matrix

        log_dict["alpha"] = avg_alpha_at_ckpt
        log_dict["step"] = step

        del noise_at_ckpt, nabla_at_ckpt
        coordinate_noise_data = {}

        if step == cfg.iterations:
            continue

        for sample_idx in sample_indexes:
            coordinate_noise_data[sample_idx] = []
        v_noise_data = []
        curr_step = step

        # inspect $sample_itr further and get the relationship between E ||g|| ~ || \nabla F ||
        for itr_step in range(1, sample_itr):
            curr_step += 1
            single_step(model, opt, scheduler, train_reader, type_ctx, distributed_backend, cfg)
            model.eval()
            noise, nabla = sample_gradient(M = M,
                                           cfg = cfg,
                                           device = device,
                                           noise_reader = noise_reader,
                                           model = model ,
                                           distributed_backend = distributed_backend ,
                                           type_ctx = type_ctx,
                                           sample_per_rank = 256)

            for sample_idx in sample_indexes:
                coordinate_noise = noise[:, sample_idx]
                norm_per_rank = torch.mean(torch.abs(coordinate_noise) ** avg_alpha_at_ckpt, dim=0)

                norm_tensor = norm_per_rank.clone().detach().to(device)
                dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
                avg_norm = (norm_tensor / distributed_backend.get_world_size()).item()
                coordinate_noise_data[sample_idx].append([torch.abs(nabla[sample_idx]).item() ** coordinate_alphas[sample_idx], avg_norm, curr_step])

            N, d = noise.shape

            noise_in_matrix : torch.Tensor = noise.reshape(N, m, n).detach().cpu()
            nabla_in_matrix = nabla.reshape(m, n)
            norm_v_per_rank = torch.mean(compute_heavy_tail_norm(noise_in_matrix, v, avg_alpha_at_ckpt), dim=0)
            norm_v_tensor = norm_v_per_rank.clone().detach().to(device)
            dist.all_reduce(norm_v_tensor, op=dist.ReduceOp.SUM)
            avg_norm_v = (norm_v_tensor / distributed_backend.get_world_size()).item()
            avg_norm_v *= nuclear_norm_of_v ** (avg_alpha_at_ckpt / 2)

            nuclear_norm_of_nabla = torch.linalg.norm(nabla_in_matrix, ord='nuc')
            v_noise_data.append([nuclear_norm_of_nabla ** avg_alpha_at_ckpt, avg_norm_v, curr_step])

            del noise, nabla, noise_in_matrix, norm_v_tensor

        if distributed_backend.is_master_process():
            world_size = dist.get_world_size()
            gather_list = [torch.empty_like(noise_norm) for _ in range(world_size)]
        else:
            gather_list = None

        dist.gather(tensor=noise_norm, gather_list=gather_list, dst=0)

        for index in coordinate_norms.keys():
            norm_at_index = coordinate_norms[index].to(device)
            if distributed_backend.is_master_process():
                world_size = dist.get_world_size()
                gather_list_coord = [torch.empty_like(norm_at_index) for _ in range(world_size)]
            else:
                gather_list_coord = None

            dist.gather(tensor=norm_at_index, gather_list=gather_list_coord, dst=0)
            if is_master:
                all_noise_norm_coord = torch.cat(gather_list_coord, dim=0)
                coordinate_norms[index] = all_noise_norm_coord
        
        # logging process
        if is_master:
            all_noise_norm = torch.cat(gather_list, dim=0)
            print("global noise acquired : {}".format(all_noise_norm.shape))
            print(f"global alpha at itr {step} : {avg_alpha_at_ckpt}")
            log_dict[f"{step_name}/distribution of noise globally"] = wandb.Table(
                data = all_noise_norm.unsqueeze(-1).tolist(),
                columns = ["norm_val"]
            )

            if n != 1:
                v_noise_map = wandb.Table(
                    data=v_noise_data,
                    columns=["nabla_norm", "noise_norm", "itr"]
                )

                log_dict[f"opt={cfg.opt}_type=v_itr={step}"] = v_noise_map

            for index in coordinate_norms.keys():
                coord_norm = coordinate_norms[index]
                log_dict[f"{step_name}/table of norm values at {index}"] = wandb.Table(
                    data = coord_norm.unsqueeze(-1).tolist(),
                    columns= ["norm_val"]
                )

            for sample_idx in sample_indexes:
                coordinate_noise_map = wandb.Table(
                    data = coordinate_noise_data[sample_idx],
                    columns = ["nabla_norm", "noise_norm", "itr"]
                )

                log_dict[f"opt={cfg.opt}_type=coordinate-{sample_idx}_itr={step}"] = coordinate_noise_map

            wandb.log(log_dict)
