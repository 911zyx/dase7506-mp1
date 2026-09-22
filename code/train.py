"""Training loop and recipe for the MP1 student submission.

Final frozen recipe (reproduces the submitted checkpoint from scratch):
  python train.py --implementation student --config configs/final.json --device cuda
                  --seed 17 --steps 24000 --batch-size 32 --eval-every 400
                  --ema-decay 0.999 --run-dir runs/final-d8gelu-24k-s17
  = 196,608,000 processed targets (~54 epochs of the 3.61M-token train split).

Recipe: AdamW betas (0.9, 0.95); weight decay 0.1 on >=2-D parameters only
(1-D norm scales exempt); lr 3e-3 with linear warmup (steps//10 unless --warmup)
and cosine decay to 10%; bf16 autocast on CUDA; grad-norm clip 1.0; EMA shadow
weights updated every step (theta_ema <- d*theta_ema + (1-d)*theta). Validation
is scored with EMA weights every --eval-every steps and the best-scoring EMA
snapshot is saved as best_checkpoint.pt (the submitted predictor); checkpoint.pt
holds the end-of-training EMA weights. --ema-decay 0 disables averaging.

Note: the baseline row in RUN_LOG.csv (baseline-s17, test 2.1013) was produced
with the ORIGINAL unmodified trainer (AdamW lr 1e-3, default betas, no EMA)
before this file was edited as permitted by the README; each run's metrics.json
records the implementation_sha256 of the code that produced it.
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score
import copy


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=0,
                   help='0=steps//10 auto')
    p.add_argument('--ema-decay', type=float, default=0.99)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    decay =[p for p in model.parameters() if p.dim()>=2]
    no_decay = [p for p in model.parameters() if p.dim()<2]
    optimizer = torch.optim.AdamW([
        {'params': decay, "weight_decay": .1}, {'params': no_decay, "weight_decay": 0.}],
        lr=args.lr, betas=(0.9, 0.95))
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best = {'bpb': float('inf'), 'step': 0}
    ema_model = copy.deepcopy(model)
    for pe in ema_model.parameters():
        pe.requires_grad_(False)
    warmup = args.warmup or max(100, args.steps // 10)
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.lr * min(1.,(step+1)/warmup) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        with torch.no_grad():
            for pe, pm in zip(ema_model.parameters(),model.parameters()):
                pe.mul_(args.ema_decay).add_(pm, alpha=1-args.ema_decay)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(ema_model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if intermediate['bpb'] < best['bpb']:
                best = {'bpb': intermediate['bpb'], 'step': step+1}
                snapshot = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
                torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                            'model':snapshot,'seed':args.seed,'ema_decay':args.ema_decay,
                            'selected_step':step+1,'train_tokens':(step+1)*args.batch_size*256},
                           args.run_dir/'best_checkpoint.pt')
                print(json.dumps({'new_best':best}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(ema_model,*data['validation'],device,'fp32')
    validation_raw=score(model, *data['validation'],device,'fp32')
    validation_raw.pop('window_nll_nats')
    validation.pop('window_nll_nats')
    if validation['bpb'] < best['bpb']:
        best = {'bpb': validation['bpb'], 'step': args.steps}
        torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                    'model':{k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()},
                    'seed':args.seed,'ema_decay':args.ema_decay,'selected_step':args.steps,
                    'train_tokens':args.steps*args.batch_size*256},
                   args.run_dir/'best_checkpoint.pt')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':ema_model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,'best_validation':best,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started, 'validation_raw':validation_raw,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
