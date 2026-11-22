"""Run each paired seed in a fresh process; retain observations and failure diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot-root', required=True)
    ap.add_argument('--engine-root', required=True)
    ap.add_argument('--fixed-work-diagnostic',action='store_true')
    ap.add_argument('--seeds', default=','.join(map(str, range(1601, 1617))))
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--opp', choices=['house', 'mix'], default='house')
    ap.add_argument('--squad', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root = Path(a.bot_root).resolve()
    engine_root = Path(a.engine_root).resolve()
    dest = Path(a.out).resolve()
    parts = dest.parent / (dest.stem + '-parts')
    parts.mkdir(parents=True, exist_ok=True)
    def run(seed):
        out = parts / f'{seed}.json'
        cmd = [sys.executable, str(HERE/'evaluate.py'), '--bot-root',str(root),
               '--engine-root',str(engine_root), '--seeds',str(seed),
               '--opp',a.opp,'--out',str(out)] + (['--squad'] if a.squad else []) + (['--fixed-work-diagnostic'] if a.fixed_work_diagnostic else [])
        env=dict(os.environ,PYTHONHASHSEED='0')
        proc = subprocess.run(cmd,cwd=root,text=True,capture_output=True,env=env)
        (parts/f'{seed}.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
        if proc.returncode:
            raise RuntimeError(f'seed {seed}: exit {proc.returncode}: {proc.stderr[-1000:]}')
        data = json.loads(out.read_text())
        row = data['matches'][0]
        print(json.dumps({'seed':seed,'points':row['points'],'late':row['late'],
                          'fallback':row['fallback_ticks']}),flush=True)
        return data
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(run,int(s)) for s in a.seeds.split(',')]
        data = [f.result() for f in as_completed(futures)]
    rows = sorted([d['matches'][0] for d in data],key=lambda r:r['seed'])
    assert len({d['summary']['sha'] for d in data}) == 1, 'Candidate changed during evaluation'
    summary = dict(data[0]['summary'])
    summary.update(seeds=a.seeds,mean_points=statistics.mean(r['points'] for r in rows),
                   max_act_p99_ms=max(r['act_ms_p99'] for r in rows))
    for k in ('late','errors','fallback_ticks','fallback_init','outside_ticks','stalled_ticks'):
        summary[k] = sum(r[k] for r in rows)
    import numpy
    metadata = {'python':platform.python_version(),'numpy':numpy.__version__,
                'fixed_work_diagnostic':a.fixed_work_diagnostic,
                'engine_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=engine_root,text=True).strip(),
                'jobs':a.jobs,'params_env':os.environ.get('AGENT_PARAMS'),
                'pythonhashseed':'0',
                'simulator_sha256':{f:hashlib.sha256((engine_root/f).read_bytes()).hexdigest()
                                    for f in ('sim/engine.py','sim/house.py')},
                'common_code_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest()
                                     for f in ('brain/nav.py','brain/world.py','brain/aim.py')},
                'bot_agent_sha256':hashlib.sha256((root/'brain/agent.py').read_bytes()).hexdigest()}
    dest.write_text(json.dumps({'summary':summary,'metadata':metadata,'matches':rows},indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
