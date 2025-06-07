# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import os

import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig, FullStateDictConfig

from verl.utils.fs import copy_local_path_from_hdfs, is_non_local

from transformers import PreTrainedTokenizer

from .checkpoint_manager import BaseCheckpointManager
import glob


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save 
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(self, model: FSDP, optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        super().__init__(model, optimizer, lr_scheduler, tokenizer)

    def load_checkpoint(self, path=None, del_local_after_load=False, *args, **kwargs):
        if path is None:
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
        remote_optim_path = os.path.join(path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
        remote_extra_state_path = os.path.join(path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')
        print(
            f'[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}'
        )
        local_model_path = copy_local_path_from_hdfs(remote_model_path)
        local_optim_path = copy_local_path_from_hdfs(remote_optim_path)
        local_extra_state_path = copy_local_path_from_hdfs(remote_extra_state_path)

        model_state_dict = torch.load(local_model_path)
        optimizer_state_dict = torch.load(local_optim_path)
        extra_state_dict = torch.load(local_extra_state_path)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(
                    f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                )

        lr_scheduler_state_dict = extra_state_dict['lr_scheduler']

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if 'rng' in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict['rng'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def clean_old_checkpoints(self, checkpoint_root: str, keep_latest: int = 1):
        """保留最新1个检查点，删除旧检查点的所有.pt文件（包括子目录）"""
        if self.rank != 0:
            return
        
        # 获取所有检查点目录并按step排序
        all_ckpt_dirs = sorted(
            glob.glob(os.path.join(checkpoint_root, "global_step_*")),
            key=lambda x: int(os.path.basename(x).split("_")[-1]),
            reverse=True
        )
        
        # 保留最新的N个目录（不处理）
        to_clean_dirs = all_ckpt_dirs[keep_latest:]
        
        for ckpt_dir in to_clean_dirs:
            try:
                # 递归删除所有.pt文件（保留其他文件）
                for root, dirs, files in os.walk(ckpt_dir):
                    # 跳过huggingface目录
                    if "huggingface" in root.split(os.sep):
                        continue
                    
                    for file in files:
                        if file.endswith(".pt"):
                            file_path = os.path.join(root, file)
                            print(f"[Clean] Removing {file_path}")
                            os.remove(file_path)
            except Exception as e:
                print(f"[Clean] Error cleaning {ckpt_dir}: {e}")

    def save_checkpoint(self, local_path: str, global_step: int, remove_previous_ckpt=True, *args, **kwargs):
        # record the previous global step
        self.previous_global_step = global_step

        # 创建目录并同步 ✅ 仅执行一次
        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                if self.optimizer is not None:
                    optimizer_state_dict = self.optimizer.state_dict()
                else:
                    optimizer_state_dict = None
                if self.lr_scheduler is not None:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler_state_dict = None

                extra_state_dict = {
                    'lr_scheduler': lr_scheduler_state_dict,
                    'rng': self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
                optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
                extra_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')

                print(f'[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}')
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_state_dict = self.model.state_dict()
        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.rank == 0:
            hf_local_path = os.path.join(local_path, 'huggingface')
            os.makedirs(hf_local_path, exist_ok=True)
            
            # First save configuration and tokenizer
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
            self.tokenizer.save_pretrained(hf_local_path)
            
            # Save the full model state dictionary
            self.model.save_pretrained(hf_local_path, state_dict=full_state_dict)

        torch.distributed.barrier()

        # 清理旧检查点时使用当前路径计算根目录 ✅
        if self.rank == 0:
            current_checkpoint_dir = os.path.dirname(local_path)
            checkpoint_root = os.path.dirname(current_checkpoint_dir)
            self.clean_old_checkpoints(checkpoint_root, keep_latest=1)
        
        # 最后记录路径 ✅
        self.previous_save_local_path = local_path
        torch.distributed.barrier()
