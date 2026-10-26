"""Run official prediction/scoring in two fresh CPU processes; external wall timing."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import argparse,hashlib,json,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'model_bundle_stack')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results_stack');args=p.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    prediction=out/f'{args.split}-predictions.json';score=out/f'{args.split}-score.json'
    commands=[[sys.executable,str(ROOT/'predict.py'),'--method',str(ROOT/'submission_stack.py'),
               '--checkpoint',str(args.checkpoint.resolve()),'--split',args.split,'--output',str(prediction)],
              [sys.executable,str(ROOT/'score_predictions.py'),'--split',args.split,
               '--predictions',str(prediction),'--output',str(score)]]
    durations=[];start=time.perf_counter()
    for command in commands:
        tick=time.perf_counter();subprocess.run(command,check=True,cwd=ROOT,env=os.environ.copy());durations.append(time.perf_counter()-tick)
    total=time.perf_counter()-start
    assets={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(args.checkpoint.iterdir()) if f.is_file()}
    cpu=platform.processor()
    if platform.system()=='Windows':cpu=subprocess.check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],text=True).strip()
    record=dict(split=args.split,wall_seconds=total,prediction_process_seconds=durations[0],scoring_process_seconds=durations[1],
                thread_limit=4,local_cpu=cpu,platform=platform.platform(),python=sys.version,assets=assets,
                asset_bytes=sum(a['bytes'] for a in assets.values()),asset_mib=sum(a['bytes'] for a in assets.values())/2**20,
                reference_machine_verified=False,commands=commands)
    (out/f'{args.split}-resources.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('assets','commands')},indent=2))
if __name__=='__main__':main()

