import os
from utils import *
from models.utils import get_model
from optim.base import train
from optim.noise import eval_noise
import math


def main(args, parser):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    exp_name = get_exp_name(args)
    exp_dir = Path(args.results_base_folder) / exp_name
    if args.mode == "train":
        run_name = "training_" + args.dataset + "_" + exp_name
    else:
        run_name = "noise_" + args.dataset + "_" +exp_name
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            entity=args.wandb_entity,
            mode="offline"
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    # print(f"Config:\n{vars(args)}\n")
    #
    # print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)


    model = get_model(args).to(
        args.device
    )  # todo: take care of initializing the model if args.use_pretrained != 'none'
    # print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs(
        config=args
    )
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    nonemb_param_cnt = (
            params_cnt
            - distributed_backend.get_raw_model(model).transformer.wpe.weight.numel()
            - distributed_backend.get_raw_model(model).transformer.wte.weight.numel()
    )
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    print("number of non-embedding parameters: %.2fM" % (nonemb_param_cnt / 1e6,))

    # if args.chinchilla:
    #     print("setting iteration steps to 1xChinchilla optimal")
    #     true_itr_steps = optimized_params_cnt /
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {
                "parameters": params_cnt,
                "optimized_parameters": optimized_params_cnt,
                "non_embedding_parameters": nonemb_param_cnt,
            }
        )

    args.world_size = distributed_backend.get_world_size()

    opt = get_opt(args, model, group_specs)
    scheduler = get_scheduler(args, opt, group_specs)
    # print(f"\nOptimizer:\n{opt}")

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            # args.resume_from = str(exp_dir / "ckpts" / "latest")
            # We did not specify a dir of resuming, considering that it should start from begin...
            # Why setting args.resume_from?
            pass
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)


    if args.mode == "train":
        stats = train(
            model=model,
            opt=opt,
            datareaders=datareaders,
            scheduler=scheduler,
            exp_dir=exp_dir,
            distributed_backend=distributed_backend,
            save_cnt=args.save_cnt,
            cfg=args,
        )
        stats["args"] = vars(args)

        if distributed_backend.is_master_process():
            with open(exp_dir / "summary.json", "w") as fs:
                json.dump(stats, fs)
    else:
        data_srcs = get_dataset(args)
        noise_reader = DataReader(
            data_src=data_srcs["train"],
            batch_size=1,
            sequence_length=args.sequence_length,
            seed=args.data_seed,
            with_replacement=False,
            auto_shard=True,
            keep_in_ram=args.data_in_ram,
        )

        eval_noise(model=model
                   , opt=opt
                   , datareaders=datareaders
                   , scheduler=scheduler
                   , cfg=args
                   , exp_dir=exp_dir
                   , save_cnt=args.save_cnt
                   , distributed_backend=distributed_backend
                   , noise_reader=noise_reader
                   )

    distributed_backend.finalize()



if __name__ == "__main__":
    Args, Parser = get_args()
    main(Args, Parser)
