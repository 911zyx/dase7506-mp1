"""Fixed scorer. Student models receive inputs, never reference next-token targets."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha, windows


@torch.no_grad()
def score(model, tokens, byte_count, device, precision, batch_size=32):
    previous_mode = model.training
    model.eval()
    started = time.perf_counter()
    nll, count, window_nll = 0., 0, []
    for x, y in windows(tokens, batch_size):
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            logp = model.predict_log_probs(x).float()
        if logp.shape != (*x.shape, 2048) or not torch.isfinite(logp).all():
            raise ValueError('Return finite log probabilities of shape [batch, time, 2048].')
        if torch.logsumexp(logp, dim=-1).abs().max().item() > 1e-3:
            raise ValueError('The output is not a normalized probability distribution.')
        losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        losses.masked_fill_(y == -100, 0)
        per_window = losses.double().sum(-1).cpu().tolist()
        window_nll.extend(per_window)
        nll += sum(per_window)
        count += (y != -100).sum().item()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    model.train(previous_mode)
    return {'bpb': nll / math.log(2) / byte_count, 'token_ppl': math.exp(nll/count),
            'nll_nats': nll, 'targets': count, 'utf8_bytes': byte_count,
            'seconds': time.perf_counter()-started, 'window_nll_nats': window_nll}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='fp32',
                   help='Main-board evaluation uses FP32 on both CPU and GPU.')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--split', choices=['validation','test'], default='test')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    device, precision = setup(args.device, args.precision, args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    model, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    data = load_data()
    result = score(model, *data[args.split], device, precision)
    losses = result.pop('window_nll_nats')
    output = args.output or args.checkpoint.parent / f'{args.split}_{device.type}_{precision}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix('.window-nll.npy'), np.asarray(losses))
    result.update(protocol=PROTOCOL, split=args.split, precision=precision,
                  checkpoint_sha256=sha(args.checkpoint), implementation_sha256=implementation_sha,
                  evaluator_sha256=sha(Path(__file__)), tokenizer_sha256=sha(ROOT/'data/tokenizer.json'),
                  **device_metrics(device))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
