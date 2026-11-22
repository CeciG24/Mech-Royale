"""Reproduce tests, official conformance, and the paired local training set."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print(' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docker',action='store_true')
    ap.add_argument('--image',default='mech-royale-review:best')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--seeds',default=','.join(map(str,range(1601,1617))))
    ap.add_argument('--jobs',type=int,default=1)
    ap.add_argument('--out',default=str(ROOT/'evaluation-retest.json'))
    a = ap.parse_args()
    if a.docker:
        run(['docker','build','--platform','linux/amd64','-t',a.image,'.'])
        sandbox = ['docker','run','--rm','-i','--platform','linux/amd64',
                   '--network','none','--cpus','1','--memory','512m','--memory-swap','512m',
                   '--pids-limit','128','--cap-drop','ALL','--read-only',
                   '--tmpfs','/tmp:rw,size=64m','--user','65534:65534',
                   '-v',str(ROOT/'mechfight-kit/arena')+':/arena:ro']
        run(sandbox+['--entrypoint','python3',a.image,'-m','unittest','discover','-s','tests','-v'])
        run([sys.executable,ROOT/'tools/check_docker.py','--map-dir','/arena','--']+sandbox+[a.image])
    else:
        run([sys.executable,'-m','unittest','discover','-s','tests','-v'])
        run([sys.executable,'mechfight-kit/check/check.py','--',sys.executable,'-u','bot.py'])
    if not a.check_only:
        run([sys.executable,'tools/batch.py','--bot-root',ROOT,'--engine-root',ROOT,
             '--squad','--seeds',a.seeds,'--jobs',str(a.jobs),'--out',a.out])


if __name__ == '__main__':
    main()
