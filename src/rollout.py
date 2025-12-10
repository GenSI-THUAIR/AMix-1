'''
Some code modified partially from ESM implementation in Huggingface and DPLM (https://github.com/bytedance/dplm).
---------------------------
Copyright (c) 2025 Institute for AI Industry Research (AIR), Tsinghua University, and AI For Science Group, Shanghai Artificial Intelligence Laboratory
SPDX-License-Identifier: Apache-2.0
'''
import json
from transformers import EsmTokenizer
import torch.nn.functional as F
from torch.utils.data import Sampler, BatchSampler, Dataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from typing import Iterable
import sys
import random 
import os
import argparse
import re
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import sys
import torch
from transformers import EsmTokenizer
import torch.nn.functional as F
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import re
import json
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable
from torch.utils.data import Sampler, BatchSampler
from tqdm import tqdm
import numpy as np
from typing import Iterable
import math
import logging
import json

tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')

class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close together."""

    def __init__(
        self, sequence_lengths: Iterable, bucket_size: int, num_replicas: int = 1, rank: int = 0
    ):
        if dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size] for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ApproxBatchSampler(BatchSampler):
    """
    Parameters:
    -----------
    sampler : Pytorch Sampler
            Choose base sampler class to use for bucketing

    max_tokens : int
            Maximum number of tokens per batch

    max_batch: int
            Maximum batch size

    sample_lengths : array-like
            List of lengths of sequences in the order of the dataset
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        drop_last=False,
        batch_size=None,
        max_len=512,
        debug_mode = False,
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.debug_mode = debug_mode
        self.batches = self._build_batches()
        
        
    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if linear <= self.max_tokens and quadratic < self.max_square_tokens:
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print('Current batch is empty! idx is ', idx)
                    continue
                batches.append(batch)
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
            if self.debug_mode and len(batches) > 150:
                break
        if len(batch) > 0:
            batches.append(batch)
            
        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            print(f'==============Local Rank {self.sampler.rank} Num Samples {num_samples}==============')
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            print(f'==============All Reduce Num Samples {num_samples}==============')
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = num_samples // len(batches)
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
            print(f'==============After Reduce, Rank{self.sampler.rank}, Num Samples {num_samples}==============')
        return batches
            
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

def discreteBayesianFlow(t, x, beta1, beta_time_order=2, mask = None):
    """
    Args:
        t: [B, N]
        x: [B, N, K], already one-hot
        beta1: [B, N]
    """
    # last_index = mask.sum(dim=-1).long() -1
    # if mask is not None:
        
    K = x.size(-1)
    beta = beta1 * (t**beta_time_order)  # (B, N)
    beta = beta.unsqueeze(-1)  # (B, N, 1)
    mean = beta * (K * x - 1)  # (B, N, K)
    std = (beta * K).sqrt()  # (B, N, 1)
    eps = torch.randn_like(mean)  # (B, N, K)
    y = mean + std * eps  # (B, N, K)
    theta = F.softmax(y, dim=-1)  # (B, N, K)
    if mask is not None:
        theta = theta * mask[...,None] + (1 - mask[...,None]) * x
    return theta


def extract_profile(rounds_idx, data):
    rounds_idx = str(rounds_idx)
    results = []
    for protein in data: 
        name = protein["sample_meta"]["id"]
        seqs = [re.sub(r"[a-z]", "", seq_info["seq"]) for seq_info in protein["rounds"][rounds_idx]]
        # print(seqs)
        # exit()
        profile = tokenizer.batch_encode_plus(seqs,add_special_tokens=True,
                                                padding="longest",
                                                return_tensors='pt')
        profile = F.one_hot(profile['input_ids'], num_classes=len(tokenizer)).sum(dim=0)
        profile[1:-1, :4] = 0
        profile[1:-1, 24:] = 0
        # print(profile)
        
        profile = (profile/profile.sum(dim=-1)[...,None])
        results.append({"name":name, "profile": profile})
    return results
        

class ProfileDataset(Dataset):
    def __init__(
        self, json_path, rounds_idx = 0, num_seqs = 50,
    ):
        data = json.load(open(json_path,'r'))
        # tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
        self.results = extract_profile(rounds_idx, data) * num_seqs
        # results = results 
        self.meta_lens = [x['profile'].shape[0] for x in self.results]
        # results =
    def __len__(self):
        return len(self.results)
    def get_metadata_lens(self):
        return self.meta_lens
    def __getitem__(self, idx):
        return self.results[idx]
    def collate(self, batch_list):
        names = [x['name'] for x in batch_list]
        # lens = [x['profile'].shape[0] for x in batch_list]
        lengths = torch.tensor(
            [x['profile'].shape[0] for x in batch_list],
            requires_grad=False,
        )
        max_L = max(lengths)
        padding_mask = torch.arange(max_L).expand(
            len(lengths), max_L
        ) < lengths.unsqueeze(1)

        profiles = pad_sequence([x['profile'] for x in batch_list],
                    batch_first=True,
                    padding_value=0,
                )
        
        batch = {
            'inputs_embeds': profiles,
            'attention_mask': padding_mask.bool(),
            'names': names
        }
        return batch

def generation(model, batch, start_t = 0.99, infer_step = 50):
        # infer_step = int((1 - self.cfg.infer_start) * self.cfg.num_diffusion_timesteps)
        # inputs_embeds.
    # print(batch)
    inputs_embeds = batch['inputs_embeds'].to(model.device).to(model.dtype)
    attention_mask = batch['attention_mask'].to(model.device)
    mask = attention_mask.clone()
    idx_mask = mask.sum(dim=-1)- 1
    mask = mask.to(model.dtype)
    mask[torch.arange(mask.shape[0]),idx_mask] = 0
    # print(mask.shape)
    mask[:,0] = 0
    probs = inputs_embeds
    for idx, _t in enumerate(tqdm(np.linspace(start = start_t, stop=1.0, num=infer_step + 1)[:-1])):
        t = (torch.ones_like(attention_mask) * _t).to(inputs_embeds)
        # if idx > 0:
        inputs_embeds = discreteBayesianFlow(t, probs, beta1=model.bfn.cfg.beta1, beta_time_order=model.bfn.cfg.beta_time_order, mask = mask)
        pred_logits = model.bfn.forward(t, inputs_embeds, attention_mask)
        probs = torch.nn.functional.softmax(pred_logits, dim=-1) * mask[..., None] + inputs_embeds * (1 - mask[..., None])
    
    probs = probs[...,1:-1,:]
    probs[...,:4]= 0
    probs[...,24:] = 0
    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(torch.argmax(probs, dim=-1), skip_special_tokens=True)
    ]
    return output_results


def get_model(ckpt_path):    
    # root_path = Path(ckpt_path).parents[1]

    # code_path = Path(root_path, "code")
    # # print(code_path)
    # sys.path.append(str(code_path))
    # cfg_path = Path(root_path, ".hydra", "config.yaml")
    # ckpt_cfg = OmegaConf.load(cfg_path)


    # ckpt_cfg.model.bfn.cfg.beta1 = ckpt_cfg.data.bfn_args.beta1
    # ckpt_cfg.model.bfn.cfg.beta_time_order = ckpt_cfg.data.bfn_args.beta_time_order
    # ckpt_cfg.model.bfn.net.config._attn_implementation='sdpa'
    # # ckpt_cfg.model.bfn.cfg.infer_start = 1.0
    # ckpt_cfg.model.test_output_dir= None
    # ckpt_cfg.model.criterion = None
    # ckpt_cfg.model.optimizer = None
    # ckpt_cfg.model.scheduler = None

    # model =  hydra.utils.instantiate(ckpt_cfg.model)
    # ckpt =torch.load(ckpt_path, map_location='cpu')
    # model.load_state_dict(ckpt['state_dict'])
    # del ckpt
    # model = model.cuda().to(torch.bfloat16).eval()
    root_path = Path(ckpt_path).parents[1]
    sys.path.append(str(root_path))
    cfg_path = Path(root_path, ".hydra", "config.yaml")
    ckpt_cfg = OmegaConf.load(cfg_path)

    ckpt_cfg.model.bfn.net.config._attn_implementation = 'sdpa'
    ckpt_cfg.model.bfn.cfg.num_diffusion_timesteps = 100

    model = hydra.utils.instantiate(ckpt_cfg.model)
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])
    model = model.cuda()
    return model

def get_dataloader(json_path, rounds_idx = 0, num_seqs = 50):
    dataset = ProfileDataset(json_path, rounds_idx, num_seqs)
    # collater = UniRefCollater(bfn_args)
    
    lens = dataset.get_metadata_lens()
    # if batch_strategy == 'lens':
    sampler = SortishSampler(lens, 1000, num_replicas=1, rank=0)

    batch_sampler = ApproxBatchSampler(sampler, 40000, 800, lens,
                                        max_len=1024)
    dl = DataLoader(
        dataset=dataset, 
        batch_sampler=batch_sampler, 
        num_workers=8, 
        pin_memory=True, 
        collate_fn=dataset.collate
    )

    return dl
      
def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # forbid random hash
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_generation(ckpt_path, json_data, round_idx=0, num_seqs=10, time=60, infer_step=50):
    """
    Run protein generation with given parameters
    
    Args:
        ckpt_path (str): Path to model checkpoint
        json_data (str): Path to JSON data file
        round_idx (int): Round index to use as source
        num_seqs (int): Number of sequences to generate
        time (int): Time parameter (0-100)
        infer_step (int): Number of inference steps
    """
    seed_everything()
    model = get_model(ckpt_path=ckpt_path)
    dataloader = get_dataloader(json_data, round_idx, num_seqs)
    results = {}
    with torch.no_grad():
        for batch in dataloader:
            
            output = generation(model, batch, start_t=time/100, infer_step=infer_step)
            for name, pred_seq in zip(batch['names'], output):
                if name in results:
                    results[name].append(pred_seq)
                else:
                    results[name] = [pred_seq]
    
    
    # Distribute sequences back to samples
    with open(json_data, "r") as f:
        data = json.load(f)
    seqs_per_sample = num_seqs
    new_round_str = str(round_idx)  # New round index as string
    for i, sample in enumerate(data):
        # Create new round if needed
        if sample["sample_meta"]["id"] in results:
            sample["rounds"][new_round_str] = [{"seq": seq,
                                                "score":{}} for seq in results[sample["sample_meta"]["id"]]]
        else:
            print(f"Warning: Sample {sample['sample_meta']['id']} not found in results. Creating empty round.")
            sample["rounds"][new_round_str] = []
    with open(json_data, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--num-seqs", type=int, default=10)
    parser.add_argument("--time", type=int, default=60)
    # parser.add_argument("--batch-size", type=int, default=1)
    # parser.add_argument("--input-a3m", type=str, required=False, help="Legacy option: Path to input A3M/FASTA file")
    # parser.add_argument("--output-json", type=str, required=False, help="Legacy option: Path to output A3M/FASTA file")
    parser.add_argument("--infer-step", type=int, default=50)
    parser.add_argument("--json-data", type=str, required=False, help="Path to JSON data file")
    parser.add_argument("--round-idx", type=int, required=False, default=0, help="Round index to use as source")
    # parser.add_argument("--new-round-idx", type=int, required=False, default=None, help="New round index for generated sequences")
    # parser.add_argument("--use-json", action="store_true", help="Use JSON data format instead of A3M/FASTA")

    args = parser.parse_args()
    
    run_generation(
        ckpt_path=args.ckpt_path,
        json_data=args.json_data,
        round_idx=args.round_idx,
        num_seqs=args.num_seqs,
        time=args.time,
        infer_step=args.infer_step
    )
