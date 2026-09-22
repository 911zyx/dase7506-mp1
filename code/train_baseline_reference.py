"""Reconstruction of original baseline recipe, not original source bytes.
Unchanged baseline model; default AdamW betas; no EMA; CPU FP32, seed17, 1200 steps.
"""
import argparse,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from common import ROOT,PROTOCOL,setup,load_data,make_model,sha
from evaluate import score
def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=17);args=p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):p.error("Use an empty run directory.")
    device,precision=setup("cpu","fp32",4);torch.manual_seed(args.seed)
    data=load_data();config=json.loads((ROOT/"configs/baseline.json").read_text())
    model,model_sha=make_model("model",config,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    tokens=data["train"][0].to(device);rng=torch.Generator().manual_seed(args.seed)
    started=time.perf_counter()
    for step in range(1200):
        starts=torch.randint(len(tokens)-257,(32,),generator=rng).to(device)
        batch=tokens[starts[:,None]+torch.arange(257,device=device)]
        lr=.001*min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*step/1200)))
        for group in optimizer.param_groups:group["lr"]=lr
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    seconds=time.perf_counter()-started
    val=score(model,*data["validation"],device,"fp32");val.pop("window_nll_nats")
    args.run_dir.mkdir(parents=True,exist_ok=True);ck=args.run_dir/"checkpoint.pt"
    torch.save(dict(protocol=PROTOCOL,implementation="model",config=config,model=model.state_dict(),seed=args.seed,train_tokens=9830400),ck)
    result=dict(validation=val,train_seconds=seconds,train_tokens=9830400,checkpoint_sha256=sha(ck),
                implementation_sha256=model_sha,note="Reconstructed recipe, not original trainer bytes.")
    (args.run_dir/"metrics.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
