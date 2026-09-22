import hashlib
import importlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
PROTOCOL = '7506-mp1-wt2-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha(ROOT / 'data' / name) != expected:
            raise ValueError(f'Changed benchmark file: {name}')
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
    data = {}
    for split in ('train', 'validation', 'test'):
        raw = (ROOT / 'data' / f'wikitext_{split}.txt').read_bytes()
        ids = tokenizer.encode(raw.decode('utf-8')).ids
        data[split] = (torch.tensor(ids, dtype=torch.long), len(raw))
    return data


def setup(device_name, precision, threads):
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(device_name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('The measured reference devices are CPU and CUDA.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(min(1., 20e9 / torch.cuda.get_device_properties(device).total_memory), device)
        torch.cuda.reset_peak_memory_stats(device)
    if precision == 'auto':
        precision = 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('Use fp32 on CPU or CUDA devices without BF16 support.')
    return device, precision


def autocast(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16')


def make_model(implementation, config, device):
    module = importlib.import_module(implementation)
    model = module.build_model(config).to(device)
    if model.context != 256 or config['vocab'] != 2048:
        raise ValueError('The main board fixes context=256 and vocab=2048.')
    return model, sha(module.__file__)


def windows(tokens, batch_size=32, context=256):
    for start in range(0, len(tokens) - 1, context * batch_size):
        chunks = [tokens[i:min(i + context + 1, len(tokens))] for i in
                  range(start, min(start + context * batch_size, len(tokens)-1), context)]
        x = torch.zeros(len(chunks), context, dtype=torch.long)
        y = torch.full_like(x, -100)
        for row, chunk in enumerate(chunks):
            x[row, :len(chunk)-1] = chunk[:-1]
            y[row, :len(chunk)-1] = chunk[1:]
        yield x, y


def device_metrics(device):
    return {
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'peak_allocated_gb': torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.,
        'peak_reserved_gb': torch.cuda.max_memory_reserved(device)/1e9 if device.type == 'cuda' else 0.,
    }
