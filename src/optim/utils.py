import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb


def get_batch(datareader, device="cpu"):
    x, y = datareader.sample_batch()
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def eval(
    model,
    reader,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
    moe=False,
    get_router_logits=False,
    cfg=None,
):
    assert model.training == False

    loss_list_val, acc_list, loss_list_aux_val = [], [], {}
    router_logits = []

    for idx in range(max_num_batches):
        x, y = get_batch(reader, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True, moe=moe)
        val_loss = outputs["loss"]

        loss_list_val.append(val_loss)
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        # auxiliary losses are optional
        for k, v in outputs["aux_losses"].items():
            loss_list_aux_val[k] = loss_list_aux_val.get(k, [])
            loss_list_aux_val[k].append(v)

        # router logits for MoE visualization
        if get_router_logits:
            # shape [layers, batch_size * sequence_length, num_experts]
            logits = outputs["router_logits"]
            # shape [max_batches, layers, batch_size * sequence_length, num_experts]
            router_logits.append(logits)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss
    val_aux_losses = {
        f"val/{k}": torch.stack(v).mean().item() for k, v in loss_list_aux_val.items()
    }

    if get_router_logits:
        # filter out the router logits that are not of the expected shape (happens for the last batch in
        # dataloader has a different batch size than the others)
        if cfg:
            intended_size = cfg.batch_size * cfg.sequence_length
        else:
            intended_size = x.shape[0] * x.shape[1]
        # shape [batches - 1, layers, batch_size * sequence_length, num_experts]
        router_logits = (
            torch.stack(
                [rl for rl in router_logits if rl.shape[1] == intended_size],
                dim=0,
            )
            .detach()
            .cpu()
        )

    return val_acc, val_loss, val_perplexity, val_aux_losses, router_logits


@torch.no_grad()
def eval_sweep_dropk(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    n_heads,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = (
        torch.linspace(0.0, 0.95, 15),
        [],
        [],
        [],
    )
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True
                )
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = (
        [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
        [],
        [],
        [],
    )
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True
                )
            nph, nh = (
                outputs["num_head_pruned_per_layer"],
                outputs["num_heads_per_layer"],
            )
            frac_heads_pruned = np.sum(nph) / np.sum(
                nh
            )  # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def save_checkpoint(model, opt, scheduler, itr, ckpt_dir: Path):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "itr": itr,
    }
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, ckpt_dir / "main.pt")


def load_checkpoint(model, opt, scheduler, ckpt_path, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    itr = ckpt["itr"]
    return itr


def save_worker_state(ckpt_dir: Path):
    # Dataloader, rng states
    worker_state = {
        "rng_torch_cpu": torch.random.get_rng_state(),
        "rng_torch_gpu": torch.cuda.get_rng_state(),
        "rng_np": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(worker_state, ckpt_dir / f"worker_{rank}.pt")


def load_worker_state(ckpt_dir: Path):
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    worker_state = torch.load(ckpt_dir / f"worker_{rank}.pt", weights_only=False)
    torch.random.set_rng_state(worker_state["rng_torch_cpu"])
    torch.cuda.set_rng_state(worker_state["rng_torch_gpu"])
    np.random.set_state(worker_state["rng_np"])
    random.setstate(worker_state["rng_python"])


def get_parameter_norms(model, order=2):
    model_norm = 0
    for p in model.parameters():
        param_data = p.detach().data
        if order == float("inf"):
            param_norm = param_data.norm(p=order)
            model_norm = max(model_norm, param_norm.item())
        else:
            param_norm = param_data.norm(p=order)
            model_norm += param_norm.item() ** order

    if order != float("inf"):
        model_norm = model_norm ** (1.0 / order)

    return model_norm


def log_prodigy_lr(opt):
    effective_lrs = []

    for group in opt.param_groups:
        d = group["d"]
        lr = group["lr"]
        if group["use_bias_correction"]:
            k = group["k"]
            beta1, beta2 = group["betas"]
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1
        effective_lr = d * lr * bias_correction
        effective_lrs.append(effective_lr)

    return effective_lrs


def visualize_routing(router_logits, extra_args):
    # router_logits: [batches, layers, batch_size * sequence_length, num_experts]
    logs = {}

    n_layers = extra_args.n_layer
    num_experts = extra_args.moe_num_experts
    num_experts_per_tok = extra_args.moe_num_experts_per_tok

    # histogram over all logits to see distribution
    logs["router/logits"] = wandb.Histogram(
        router_logits.type(torch.float32).flatten().cpu().numpy()
    )

    # distribution over experts for layer 0, layer n/2, n-1
    for layer in [0, n_layers // 2, n_layers - 1]:
        router_logits_layer = router_logits[:, layer]
        # shape [batches, batch_size * sequence_length, num_experts_per_tok]
        weights, selected_experts = torch.topk(
            router_logits_layer, num_experts_per_tok, dim=-1
        )
        # shape [batches, batch_size * sequence_length, num_experts_per_tok, num_experts]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
        # For a given token, determine if it was routed to a given expert.
        # Shape: [batches, batch_size * sequence_length, num_experts]
        expert_mask, _ = torch.max(expert_mask, dim=-2)
        # shape [num_experts]
        tokens_per_expert = torch.mean(expert_mask, dim=(0, 1), dtype=torch.float32)
        layer_token_routing = {
            f"router/layer_{layer}_expert_{i}_selection": tokens_per_expert[i].item()
            for i in range(num_experts)
        }
        logs.update(layer_token_routing)
    return logs

# functions for noise analysis

def gather_alpha(local_alpha, distributed_backend, device) -> float:
    alpha_tensor = local_alpha.clone().detach().to(device)
    dist.all_reduce(alpha_tensor, op=dist.ReduceOp.SUM)
    avg_alpha = (alpha_tensor / distributed_backend.get_world_size()).item()
    return max(1.0, avg_alpha)

def get_attn_gradient(model):
    target_param = None
    layer_indices = []
    attn_params = []
    for name, param in model.named_parameters():
        if "c_attn" in name and param.grad is not None:
            parts = name.split('.')
            if len(parts) >= 3 and parts[1] == 'h':
                layer_idx = int(parts[2])
                layer_indices.append(layer_idx)
                attn_params.append((layer_idx, param))

    layer_indices.sort()
    mid_idx = len(layer_indices) // 2
    target_layer_idx = layer_indices[mid_idx]

    for layer_idx, param in attn_params:
        if layer_idx == target_layer_idx:
            target_param = param
            break

    if target_param is None:
        raise ValueError("Could not find attention layer param with active gradients.")

    return target_param.grad.detach()

def get_embd_gradient(model, device, MAX_ROWS=4096):
    target_param = None
    for name, param in model.named_parameters():
        # 'wte' stands for Word Token Embeddings
        # We check for .weight to ensure we aren't picking up bias or other sub-params
        if "transformer.wte.weight" in name and param.grad is not None:
            target_param = param
            break
    if target_param is None:
        raise ValueError("Could not find embedding parameters (wte) with active gradients.")
    V, d = target_param.shape
    g = torch.Generator(device='cpu')
    g.manual_seed(42)
    num_to_sample = min(MAX_ROWS, V)
    sample_indices = torch.randperm(V, generator=g)[:num_to_sample].to(device)
    grad = torch.index_select(target_param.grad, 0, sample_indices).detach()
    return grad.detach()

def get_gradient(cfg, model, device) -> torch.Tensor:
    """
    Targets the Word Token Embedding (WTE) layer.
    In GPT-2, this is typically model.transformer.wte.weight.
    """
    if cfg.layer_type == "attention":
        return get_attn_gradient(model)
    elif cfg.layer_type == "embedding":
        return get_embd_gradient(model, device, MAX_ROWS=2048)
    else:
        raise NotImplementedError("Layer type not implemented")

# Corollary 2.4 in Mohammadi 2014
def alpha_estimator(X) -> torch.Tensor:
    # X is N by d matrix
    N, d = X.shape
    m = 1
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    if m == 1:
        raise ValueError
    n = N // m
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff


def grad_shape(cfg, model, noise_reader, device, type_ctx, distributed_backend) -> torch.Size:
    dummy_x, dummy_y = get_batch(noise_reader, device=device)
    model.zero_grad(set_to_none=True)
    with type_ctx:
        model(dummy_x, targets=dummy_y)["loss"].backward()

    raw_model = distributed_backend.get_raw_model(model)
    ori_shape = get_gradient(cfg, raw_model, device).shape
    model.zero_grad(set_to_none=True)
    return ori_shape

def sample_gradient(cfg, M, device, noise_reader, model, type_ctx, distributed_backend, sample_per_rank = 128) -> tuple[torch.Tensor, torch.Tensor]:
    local_samples = []
    true_grad_acc = torch.zeros(M, device=device)
    batch_count = noise_reader.num_batches()
    batch_per_rank = batch_count // dist.get_world_size()
    keep_grad = min(sample_per_rank, batch_per_rank)

    for i in range(keep_grad):
        model.zero_grad(set_to_none=True)

        with model.no_sync():
            data = get_batch(noise_reader, device=device)
            x, y = data

            with type_ctx:
                loss = model(x, targets=y)["loss"]
            loss.backward()

        with torch.no_grad():
            raw_model = distributed_backend.get_raw_model(model)
            current_grad = get_gradient(cfg, raw_model, device).reshape(-1)

            # true grad
            true_grad_acc.add_(current_grad)
            local_samples.append(current_grad.cpu())

    dist.all_reduce(true_grad_acc, op=dist.ReduceOp.SUM)
    total_batches = keep_grad * dist.get_world_size()
    true_mean_grad = (true_grad_acc / total_batches)
    local_grads_tensor = torch.stack(local_samples)
    noise = local_grads_tensor - true_mean_grad.cpu()

    model.zero_grad(set_to_none=True)
    return noise, true_mean_grad

def single_step(model, opt, scheduler, train_reader, type_ctx, distributed_backend, cfg) -> None:
    model.train()
    for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
        x, y = get_batch(train_reader, device=cfg.device)
        with type_ctx:
            with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
            ):
                outputs = model(x, targets=y, moe=cfg.moe)

        loss = outputs["loss"] / cfg.acc_steps
        loss.backward()
    if cfg.grad_clip != 0.0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            torch.nn.utils.clip_grad_norm_(
                model.module.parameters(), cfg.grad_clip
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )

    if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
        opt.train()
    opt.step()
    if cfg.scheduler != "none":
        scheduler.step()

    opt.zero_grad(set_to_none=True)

    return

def produce_v(noise, p, eps=1e-8):
    """
    Produces v as an m x n rectangular diagonal matrix.
    Strictly: v[i, j] = 0 if i != j.
    """
    B, m, n = noise.shape
    # To find the 'diagonal' scale, we look at the p-th moment of the
    # diagonal elements of the noise across the batch.
    # Note: We only need the first min(m, n) elements for a diagonal v.
    k = min(m, n)

    # Extract diagonal elements of each noise matrix in the batch: (B, k)
    diag_noise = torch.diagonal(noise, dim1=1, dim2=2)

    # Calculate the p-th root of the p-th moment for the diagonal
    v_values = torch.mean(torch.abs(diag_noise).pow(p), dim=0).pow(1 / p) + eps

    # Construct the m x n matrix v
    v = torch.zeros((m, n), device=noise.device)
    indices = torch.arange(k)
    v[indices, indices] = v_values

    return v


def compute_heavy_tail_norm(noise, v, p):
    """
    Computes || G - nabla ||_{|v|^-1}^p
    where |v|^-1 = (v @ v.T)^-1
    and ||X||_L = sqrt(tr(X.T @ L @ X))
    """
    # 1. Calculate |v| = v @ v.T (m x m)
    # Since v is rectangular diagonal, v @ v.T is an m x m diagonal matrix.
    # (v @ v.T)_{ii} = v_{ii}^2 for i < min(m, n), else 0.
    v_sq_diag = torch.sum(v ** 2, dim=1) + 1e-12  # (m,) - sum along rows

    # 2. Inverse L = (v @ v.T)^-1/2
    L_diag = 1.0 / torch.sqrt(v_sq_diag)  # (m,)

    # 3. Compute tr(X.T @ L @ X)
    # tr(X.T @ L @ X) = sum_{j=1}^n sum_{i=1}^m X_ij^2 * L_ii
    # We use broadcasting to keep this fast on CPU.
    sq_noise = noise.pow(2)  # (B, m, n)

    # weighted_sq: (B, m, n) * (1, m, 1)
    weighted_sq = sq_noise * L_diag.view(1, -1, 1)

    # Trace is the sum over m and n
    trace_vals = torch.sum(weighted_sq, dim=(1, 2))  # (B,)

    # 4. Return the p-th power of the norm: (sqrt(trace))^p
    return torch.abs(trace_vals).pow(p / 2)


def nuclear_norm_of_positive_diagonal_matrix(V):
    if V.ndim != 2:
        raise ValueError("Input must be a 2D tensor.")
    diag_vals = torch.diagonal(V)  # shape: (k,)

    if not torch.all(diag_vals > 0):
        raise ValueError("All diagonal elements must be strictly greater than 0.")

    return diag_vals.sum()