"""Frozen-predictor audit; never trains or selects a new predictor."""
import argparse, ctypes, hashlib, json, os, platform, runpy, statistics, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def peak_ram():
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    m=PMC();m.cb=ctypes.sizeof(m)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(m),m.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'peak_working_set_bytes':m.PeakWorkingSetSize,'peak_commit_bytes':m.PeakPagefileUsage,
            'method':'Windows GetProcessMemoryInfo; lifetime high-water marks including imports, loading, tokenization and scoring'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--checkpoint');p.add_argument('--output');args=p.parse_args()
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    if args.child:
        sys.argv=['evaluate.py','--checkpoint',args.checkpoint,'--split','test','--device','cpu','--precision','fp32','--threads','4','--output',args.output]
        runpy.run_path(str(ROOT/'evaluate.py'),run_name='__main__')
        Path(args.output+'.ram.json').write_text(json.dumps(peak_ram(),indent=2))
        return
    out=ROOT/'submission_evidence';out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    protected=['common.py','evaluate.py','model.py','requirements.txt','tests/test_contract.py','configs/baseline.json']+[str(f.relative_to(ROOT)).replace('\\','/') for f in (ROOT/'data').iterdir() if f.is_file()]
    checks={f:sha(ROOT/f)==manifest['code/'+f] for f in protected}
    assert all(checks.values()),checks
    final=ROOT/'runs/final-d8gelu-24k-s17/best_checkpoint.pt'
    original=json.loads((final.parent/'test_cpu_fp32.json').read_text())
    assert sha(final)==original['checkpoint_sha256']
    assert sha(ROOT/'student.py')==original['implementation_sha256']
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],capture_output=True,text=True)
    (out/'contract_tests.txt').write_text(test.stdout+test.stderr)
    assert test.returncode==0
    # Run the same behavioral tests against the ACTUAL final config, not just the default small SwiGLU config.
    import torch, unittest
    from student import build_model
    sys.path.insert(0,str(ROOT/'tests'))
    from test_contract import ContractTests
    config=json.loads((ROOT/'configs/final.json').read_text())
    class FinalContract(ContractTests):
        def setUp(self):
            torch.set_num_threads(4);torch.manual_seed(17)
            self.model=build_model(config).eval()
    with (out/'final_config_contract_tests.txt').open('w') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalContract))
    assert result.wasSuccessful()
    import gc
    gc.collect()
    results={}
    for label,checkpoint in [('baseline',ROOT/'runs/baseline/checkpoint.pt'),('final',final)]:
        rows=[]
        for i in range(1,4):
            target=out/f'{label}_test_repeat{i}.json'
            with target.with_suffix('.stdout.txt').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child','--checkpoint',str(checkpoint),'--output',str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
            row=json.loads(target.read_text()); row.update(json.loads(Path(str(target)+'.ram.json').read_text()));rows.append(row)
            print(label,i,'bpb',row['bpb'],'seconds',row['seconds'],'peak GiB',row['peak_working_set_bytes']/2**30,flush=True)
        results[label]=rows
    final_rows=results['final'];base_rows=results['baseline']
    assert all(abs(x['bpb']-original['bpb'])<1e-7 for x in final_rows)
    summary={'protected_hash_checks':checks,'frozen_checkpoint_sha256':sha(final),'student_sha256':sha(ROOT/'student.py'),
      'original_test_bpb':original['bpb'],'tests':'5 default and 5 final-config contract tests passed',
      'platform':platform.platform(),'python':sys.version,'threads':4,'precision':'fp32','repeats':results,
      'baseline_median_seconds':statistics.median(x['seconds'] for x in base_rows),
      'final_median_seconds':statistics.median(x['seconds'] for x in final_rows),
      'final_peak_ram_gib':max(x['peak_working_set_bytes'] for x in final_rows)/2**30,
      'final_peak_commit_gib':max(x['peak_commit_bytes'] for x in final_rows)/2**30,
      'checkpoint_bytes':final.stat().st_size,'date_local':time.strftime('%Y-%m-%d')}
    summary['time_ratio']=summary['final_median_seconds']/summary['baseline_median_seconds']
    summary['resource_pass']=summary['time_ratio']<=5 and summary['final_peak_ram_gib']<=4 and final.stat().st_size<=64*2**20
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['repeats','protected_hash_checks']},indent=2),flush=True)
    assert summary['resource_pass']
if __name__=='__main__':main()
