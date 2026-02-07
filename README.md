# Heavy-tailed-noise-expriment
This is the experiment code for `Sign-based Optimizers Are Effective Under Heavy-tailed Noise`

## Quickstart 

To reproduce our results, you can follow the routine here:

1. Create a conda environment and install dependencies:

```sh
conda create -n env python=3.10
conda activate env
pip install -r requirements.txt
```

2. Get the C4 dataset ready for training

```sh
python ./src/data/c4.py
```

Training nanoGPT with all the optimizers:

```sh
./scripts/train_all.sh
```

Analyse the noise and gradient during training:
```sh
./scripts/noise.sh
```


## Parameters

Here are the possible parameters you can use (copypasted from `config/base.py`):

```python
# --- training --- #
# General training params
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--acc_steps', default=4, type=int)
parser.add_argument('--seed', default=0, type=int) # random seed for the parameters
parser.add_argument('--data_seed', default=1337, type=int) # random seed defining the data ordering
parser.add_argument('--eval_interval', default=200, type=int)
parser.add_argument('--full_eval_at', nargs="+", type=int)
parser.add_argument('--eval_batches', default=64, type=int)
parser.add_argument('--device', default='cuda:0', type=str) # see below to run on multiple GPUs
parser.add_argument('--iterations', default=25000, type=int) # total number of training iterations
parser.add_argument('--warmup_steps', default=300, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--wsd_final_lr_scale', default=0.0, type=float) # wsd scheduler
parser.add_argument('--wsd_fract_decay', default=0.1, type=float) # wsd scheduler 
parser.add_argument('--decay_type', default='linear', choices=['linear', 'cosine', 'exp', 'miror_cosine', 'square', 'sqrt'])
parser.add_argument('--weight_decay', default=0.1, type=float) # I recommend you keep this value, else instabilities might arise
parser.add_argument('--beta1', default=0.9, type=float) # adam parameter
parser.add_argument('--beta2', default=0.95, type=float) # adam parameter
parser.add_argument('--scheduler', default='cos', choices=['linear', 'cos', 'wsd', 'cos_inf', 'none'])
parser.add_argument('--final_div_factor', default=1, type=float) # cosine and linear schedulers
parser.add_argument('--cos_inf_steps', default=0, type=int) # cos_inf scheduler
parser.add_argument('--opt', default='adamw', choices=['adamw', 'sgd', 'muon', 'soap', 'ademamix', 'lion', 'sf-adamw', 'sf-sgd', 'signsgd', 'signum', 'prodigy', 'sophiag', 'adopt', 'mars', 'adafactor', 'lamb', 'scion', 'scion-light', 'd-muon', 'muon-pytorch'])
parser.add_argument('--eval_freq', default=200, type=int) # in iterations
parser.add_argument('--results_base_folder', default="./exps", type=str) # where the checkpoints will be saved
parser.add_argument('--grad_clip', default=0.0, type=float) # default value is 1.0 in nanoGPT
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--shampoo_beta', default=-1.0, type=float)
parser.add_argument('--precondition_frequency', default=10, type=int) #for SOAP and Sophia
parser.add_argument('--max_precond_dim', default=10000, type=int)
parser.add_argument('--merge_dims', default=False, type=bool) # merge dimensions till the product of the dimensions is less than or equal to max_precond_dim
parser.add_argument('--precondition_1d', default=False, type=bool)
parser.add_argument('--normalize_grads', default=False, type=bool)
parser.add_argument('--soap_data_format', default='channels_first', type=str)
parser.add_argument('--correct_bias', default=True, type=bool)
parser.add_argument('--nesterov', default=False, type=bool) # whether to use Nesterov-style momentum 
parser.add_argument('--muon_ns_steps', default=5, type=int) # the number of steps to use in the newton schulz, if it is iterative
parser.add_argument('--muon_lr_factor', default=0.02, type=float) # a factor by which to reduce the lr for muon
parser.add_argmunet('--adema_beta3', default=0.9, type=float) # beta3 in AdEMAMix
parser.add_argument('--adema_alpha', default=2.0, type=float) # alpha in AdEMAMix
parser.add_argument('--adema_beta3_warmup', default=None, type=int) # AdEMAMix hyperparameter
parser.add_argument('--adema_alpha_warmup', default=None, type=int) # AdEMAMix hyperparameter
parser.add_argument('--schedulefree_r', defalut=0.0, type=float) # schedulefree hyperparameter
parser.add_argument('--weight_lr_power', default=2.0, type=float) # schedulefree hyperparameter
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--dampening', default=0.0, type=float)
parser.add_argument('--prodigy_beta3', default=None, type=float) # coefficients for computing the Prodidy stepsize using running averages
parser.add_argument('--prodigy_decouple', default=True, type=bool) # Use AdamW style decoupled weight decay
parser.add_argument('--prodigy_use_bias_correction', default=False, type=bool)
parser.add_argument('--prodigy_safeguard_warmup', default=False, type=bool) # Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.
parser.add_argument('--prodigy_fsdp_in_use', default=False, type=bool)
parser.add_argument('--sophia_rho', default=0.04, type=float)
parser.add_argument('--mars_type', default='mars-adamw', choices=['mars-adamw', 'mars-lion', 'mars-shampoo'],)
parser.add_argument('--mars_vr_gamma', default=0.025, type=float)
parser.add_argument('--mars_is_approx', default=True, type=float)
parser.add_argument('--mars_lr', default=3e-3, type=float)
parser.add_argument('--mars_beta1', default=0.95, type=float)
parser.add_argument('--mars_beta2', default=0.99, type=float)
parser.add_argument('--adafactor_decay_rate', default=-0.8, type=float)
parser.add_argument('--lamb_use_bias_correction', default=False, type=bool)
parser.add_argument('--adopt_decouple', default=True, type=bool)
parser.add_argument('--adopt_eps', default=1e-6, type=float)
parser.add_argument('--scion_lmh_scale', default=10.0, type=float)
parser.add_argument('--scion_emb_scale', default=1.0, type=float)
parser.add_argument('--scion_tr_scale', default=3.0, type=float)
parser.add_argument('--weight_average', action='store_true') # uniform weight averaging (or SWA)
parser.add_argument('--wa_interval', default=5, type=int, help='How often to take the average (every k steps). Must divide wa-horizon.')
parser.add_argument('--wa_horizon', default=500, type=int, help='How frequently we save uniform model averages. Should divide '
+ 'latest-ckpt-interval, otherwise some points may not be saved ' + 'correctly.')
parser.add_argument('--wa_dtype', default='float32', type=str, choices=['float32', 'float64'])
parser.add_argument('--wa_use_temp_dir', action='store_true')
parser.add_argument('--wa_sweep_horizon', action='store_true')
parser.add_argument('--max_num_wa_sweeps', default=5, type=int)
parser.add_argument('--exponential_weight_average', action='store_true') # EMA of weights
parser.add_argument('--ewa_interval', default=10, type=int, help='How often to take the EWA average (every k steps).')
parser.add_argument('--ewa_decay', default=0.95, type=float, help='EWA decay parameter (between 0.9 and 1).')
parser.add_argument('--ewa_after_warmup', action='store_true', help='Start EWA after warmup steps.')
# Dataset params
parser.add_argument('--dataset', default='slimpajama', choices=['slimpajama', 'wikitext', 'shakespeare-char', 'arxiv', 'arxiv2000', 'arxiv+wiki', 'openwebtext2', 'redpajama', 'redpajamav2', 'fineweb', 'finewebedu', 'c4', 'arc_easy', 'arc_challenge', 'hellaswag', 'logiqa', 'piqa', 'sciq', 'humaneval', 'gsm8k', 'kodcode', 'mathqa', 'medqa'])
parser.add_argument('--tokenizer', default='gpt2', type=str, choices=['gpt2', 'mistral'])
parser.add_argument('--vocab_size', default=50304, type=int)
parser.add_argument('--data_in_ram', action='store_true') # force the data to RAM, you most likely do not need this  
# Model params
parser.add_argument('--model', default='base', choices=['base', 'llama', 'mup_gpt', 'mup_llama',])
parser.add_argument('--parallel_block', action='store_true')
parser.add_argument('--use_pretrained', default='none', type=str) # 'none', 'gpt2' or a path to the pretraind model
parser.add_argument('--from_dense', action='store_true')
parser.add_argument('--init_std', default=0.02, type=float)
parser.add_argument('--dropout', default=0.0, type=float) # keep to 0 unless in low data regime (e.g. wikitext)
parser.add_argument('--n_head', default=12, type=int)
parser.add_argument('--n_layer', default=12, type=int) # depth in (att + ff) blocks
parser.add_argument('--n_embd', default=768, type=int) # hidden size ... 
parser.add_argument('--sequence_length', default=512, type=int)
parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'float16', 'bfloat16'],)
parser.add_argument('--bias', default=False, type=bool)
parser.add_argument('--compile', action='store_true') # if true then model is compiled
parser.add_argument('--untied_embeds', action='store_true') # disables weight tying between lm_head.weight and wte.weight
parser.add_argument('--rmsnorm_eps', default=1e-5, type=float) # used by the llama model
parser.add_argument('--multiple_of', default=256, type=int) # used by the llama model make SwiGLU hidden layer size multiple of large power of 2
parser.add_argument('--moe', action='store_true')
parser.add_argument('--moe_routing', default='standard_gating', type=str, choices=['standard_gating', 'expert_choice'],)
parser.add_argument('--moe_num_experts', default=8, type=int)
parser.add_argument('--capacity_factor', default=2.0, type=float) # only used for expert choice routing
parser.add_argument('--moe_num_shared_experts', default=0, type=int) # deepseek routing, experts that are always active
parser.add_argument('--moe_router_loss', default='load_balancing_z_loss', type=str, choices=['entropy', 'load_balancing_only', 'load_balancing_z_loss'],)
parser.add_argument('--moe_num_experts_per_tok', default=2, type=int)
parser.add_argument('--moe_entropy_loss_factor', default=0.01, type=float)
parser.add_argument('--moe_aux_loss_factor', default=0.1, type=float)
parser.add_argument('--moe_z_loss_factor', default=0.01, type=float)
parser.add_argument('--moe_softmax_order', type=str, default='topk_softmax', choices=['softmax_topk', 'topk_softmax'],)
parser.add_argument('--plot_router_logits', action='store_true')
parser.add_argument('--scale_emb', default=10, type=int) # mup arguments --- the base model width that mup has been configured on
parser.add_argument('--scale_base_model', default=256, type=int)
parser.add_argument('--scale_depth', default=1.4, type=float)
# Checkpointing
parser.add_argument('--results_base_folder', default='./exps', type=str)
parser.add_argument('--permanent_ckpt_interval', default=0, type=int)
parser.add_argument('--latest_ckpt_interval', default=0, type=int)
parser.add_argument('--resume_from', default=None, type=str)
parser.add_argument('--resume_from_swa', default=None, type=str)
parser.add_argument('--auto_resume', default=True)
# logging params (WandB)
parser.add_argument('--wandb', action='store_true') # whether to use wandb or not
parser.add_argument('--wandb_project', default='my-project', type=str)
parser.add_argument('--wandb_entity', default=None, type=none_or_str) # for the team projects
parser.add_argument('--wandb_run_prefix', default='none', type=str) # is added before the autogenerated experiment name
parser.add_argument('--eval_seq_prefix', default="Once upon a time", type=str) # prefix used to generate sequences
parser.add_argument('--log_dynamics', action='store_true')
parser.add_argument('--dynamics_logger_cfg', default='./src/logger/rotational_logger.yaml', type=str)
parser.add_argument('--log_parameter_norms', action='store_true') # logs the L2 norm of the parameters
parser.add_argument('--norm_order', default=2) # order of the model norm to log
# Distributed args
parser.add_argument('--distributed_backend', default=None, type=str, required=False,
                    choices=distributed.registered_backends())  # distributed backend type (e.g. nccl)
# --- noise --- #
parser.add_argument("--mode", default="train", type=str) # training or noise analyse
parser.add_argument("--save_cnt", default=5, type=int) # Count of ckpts saved during training
parser.add_argument("--sample_itr", default=50, type=int) # How many iterations you want to test at each ckpt
parser.add_argument("--layer_type", default="attn", type=str) # which layer's gradient to fetch, attention or embedding
```

## Using WandB

The project relies on wandb to save the outcome and record the results. You need to give your wandb authorize key in order to send the data to your wandb account. If you start jobs on a server without access to prompt, then you can set the `WANDB_API_KEY` variable within your script:

```bash
# this is a script that could be executed on a server
pip install -r requirements.txt # install req.
export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
```

## Acknowledgements

This project is heavily based on [llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark).

The tail-index estimator `alpha_estimator` is borrowed from [sgd_tail_index](https://github.com/umutsimsekli/sgd_tail_index)

We gratefully acknowledge their work and contributions to the open-source community.