import os
import copy
import random
import sys
import hydra
import submitit
from omegaconf import DictConfig

from trainer.accelerators.utils import nvidia_smi_gpu_memory_stats, print_config

DEEPSPEED_MULTINODE = "<is_deepspeed_multinode>"

def print_env():
    for key in sorted(os.environ.keys()):
        if not (
                key.startswith(("SLURM_", "SUBMITIT_"))
                or key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        ):
            continue
        value = os.environ[key]
        print(f"{key}={value}")


class Task:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self):
        print("Running task on slurm")
        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment()
        rng = random.Random(dist_env._job_env.job_id)
        dist_env.master_port = rng.randint(10000, 20000)
        dist_env = dist_env.export()
        os.environ.update(**{
            "CUDA_LAUNCH_BLOCKING": "0",
            "NCCL_DEBUG": "info",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "XDG_CACHE_HOME": "./.cache/",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
        })
        print(nvidia_smi_gpu_memory_stats())
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        print("Running training script")
        print(f"Local rank {dist_env.local_rank}: {os.environ['CUDA_VISIBLE_DEVICES']=}")
        num_processes = self.cfg.slurm.n_processes * self.cfg.slurm.n_nodes
        # num_processes = self.cfg.slurm.n_processes
        machine_rank = dist_env.rank // self.cfg.slurm.n_processes

        cmd = f"accelerate launch --dynamo_backend no --gpu_ids all --mixed_precision {self.cfg.slurm.mixed_precision} --num_processes {num_processes} {DEEPSPEED_MULTINODE} --num_machines {self.cfg.slurm.n_nodes} --use_deepspeed --machine_rank {machine_rank} --main_process_ip {dist_env.master_addr} --main_process_port {dist_env.master_port} trainer/scripts/train.py {self.cfg.slurm.cmd}"

        if self.cfg.slurm.n_nodes > 1:
            hostfile_dir = "hostfiles"
            os.makedirs(hostfile_dir, exist_ok=True)
            hostfile = os.path.realpath(f"{hostfile_dir}/{dist_env._job_env.job_id}.txt")
            if dist_env.rank == 0:
                with open(hostfile, "w") as f:
                    for host in dist_env._job_env.hostnames:
                        f.write(f"{host} slots={self.cfg.slurm.n_processes}\n")
                print(f"Created hostfile: {hostfile}")
            cmd = cmd.replace(DEEPSPEED_MULTINODE, f"--deepspeed_hostfile {hostfile} --deepspeed_multinode_launcher standard")
        else:
            cmd = cmd.replace(DEEPSPEED_MULTINODE, "")

        if dist_env.local_rank == 0:
            print(f"Running command: {cmd}")
            exit_code = os.system(cmd)
        else:
            exit_code = 0
            print("Waiting for master to finish")
        if exit_code != 0:
            raise RuntimeError(f"Command {cmd} failed with exit code {exit_code}")

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)


@hydra.main(version_base=None, config_path="../conf", config_name="slurm_config")
def main(cfg: DictConfig) -> None:
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=5900, stdoutToServer=True, stderrToServer=True)
    executor = submitit.AutoExecutor(folder="logs")
    print_config(cfg)

    slurm_additional_parameters = {
        # "gpus": cfg.slurm.n_processes,
        "ntasks_per_node": cfg.slurm.n_processes,
        "gpus_per_node": cfg.slurm.n_processes,
        
    }

    if cfg.slurm.account is not None:
        slurm_additional_parameters["account"] = cfg.slurm.account

    print(f"SLURM additional parameters: {slurm_additional_parameters}")
    
    slurm_kwargs = {
        "slurm_job_name": cfg.slurm.job_name,
        "slurm_partition": cfg.slurm.partition,
        "slurm_nodes": cfg.slurm.n_nodes,
        "slurm_additional_parameters": slurm_additional_parameters,
        "slurm_cpus_per_task": 12,
        "slurm_time": cfg.slurm.time_limit,
        "slurm_exclude": cfg.slurm.exclude if cfg.slurm.exclude else "",
        "stderr_to_stdout": True,
        "slurm_array_parallelism": cfg.slurm.array_parallelism
        # "slurm_mem": "50GB", # not needed for our cluster
    }
    executor.update_parameters(**slurm_kwargs)
    
    
    jobs = []
    with executor.batch():
        print("Submitting jobs")
        # for seq_length in (4,):
        for dp in (0.2, 0.4, 0.6, 0.8):
            cur_cfg = copy.deepcopy(cfg)
            cur_cfg.slurm.cmd += f" run_name=data_prop_{dp} \
                            dataset.sequence_length=6 \
                            dataset.data_prop={dp}"
            task = Task(cur_cfg)
            job = executor.submit(task)
            jobs.append(job)

                
                
        
        
        
        # for seq_length in (4,5,6,7,8,9):
        #     # for lr in (1e-4,):
        #     for lr in (1e-4, 5e-5, 1e-5, 5e-6):
        #         # for warmup_steps in (0,):
        #         for warmup_steps in (0,500,1000):
        #             cur_cfg = copy.deepcopy(cfg)
        #             cur_cfg.slurm.cmd += f" run_name=seq_length_{seq_length}_lr_{lr}_warmup_{warmup_steps} \
        #                             dataset.sequence_length={seq_length} \
        #                             accelerator.deepspeed.optimizer.params.lr={lr} \
        #                             accelerator.deepspeed.scheduler.params.warmup_max_lr={lr} \
        #                             accelerator.deepspeed.scheduler.params.warmup_num_steps={warmup_steps}"
        #             # cur_cfg.experiment.run_name = f"seq_length_{seq_length}_lr_{lr}_warmup_{warmup_steps}"
        #             # cur_cfg.experiment.dataset.sequence_length = seq_length
        #             # cur_cfg.experiment.deepspeed.optimizer.params.lr = lr
        #             # cur_cfg.experiment.deepspeed.scheduler.params.warmup_max_lr = lr
        #             # cur_cfg.experiment.deepspeed.scheduler.params.warmup_num_steps = warmup_steps
        #             task = Task(cur_cfg)
        #             job = executor.submit(task)
        #             jobs.append(job)
    if len(jobs) > 0:
        print(f"Tracking {len(jobs)} jobs")
        submitit.helpers.monitor_jobs(jobs)
    else:
        print("No jobs to track")
        
    print("Done")


if __name__ == "__main__":
    sys.exit(main())
