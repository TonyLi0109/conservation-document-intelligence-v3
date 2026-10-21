"""Resume one stopped model arm without changing a concurrent coordinator's manifest."""
import argparse
from pathlib import Path
import sys
import time
import subprocess
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=run.METHODS, required=True)
    parser.add_argument('--not-before', help='ISO timestamp with timezone for a subscription reset')
    parser.add_argument('--export-after', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.not_before:
        target = datetime.fromisoformat(args.not_before)
        if target.tzinfo is None:
            raise ValueError('Scheduled time requires an explicit timezone')
        scheduled = root/'runs'/args.method/'scheduled_resume.json'
        run.save(scheduled, {'status':'waiting_for_subscription_reset','scheduled_at':target.isoformat(), 'route':'existing subscription'})
        while (target-datetime.now(timezone.utc)).total_seconds() > 0:
            time.sleep(min(30, max(0.01,(target-datetime.now(timezone.utc)).total_seconds())))
        run.save(scheduled, {'status':'resuming','scheduled_at':target.isoformat(),'started_at_utc':run.now()})
    run.verify(root)
    folder = root/'runs'/args.method
    states = [run.read(p) for p in folder.glob('Q*/status.json')]
    if any(s.get('status') == 'running' for s in states):
        raise RuntimeError('Arm already marked running; refusing concurrent writes.')
    previous = folder/'failure.json'
    if previous.exists():
        history = folder/'resume_history'
        history.mkdir(exist_ok=True)
        previous.rename(history/f"failure_{len(list(history.glob('failure_*.json')))+1}.json")
    status = folder/'resume_status.json'
    run.save(status, {'status':'running', 'started_at_utc':run.now(),
                      'policy':'Unchanged frozen runner, prompts, validators, model and subscription route; accepted calls reused; failed calls get new numbered attempts.'})
    try:
        run.run_method(root,args.method)
    except Exception as error:
        run.save(folder/'failure.json', {'at':run.now(),'error':str(error)})
        run.save(status, {'status':'failed','finished_at_utc':run.now(),'error':str(error)})
        raise
    run.save(status, {'status':'completed','finished_at_utc':run.now()})
    if args.not_before:
        run.save(scheduled, {'status':'completed','finished_at_utc':run.now()})
    if args.export_after:
        with (root/'handoff_export_after_resume.log').open('w',encoding='utf-8') as log:
            result = subprocess.run([sys.executable,str(run.ROOT/'scripts/export_corrected_pdf_handoff.py'),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
        run.save(root/'handoff_export_status.json',{'status':'completed' if result.returncode == 0 else 'incomplete_or_export_error', 'exit_code':result.returncode,'updated_at_utc':run.now()})


if __name__ == '__main__':
    main()
