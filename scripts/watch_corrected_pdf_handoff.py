"""Wait for a running benchmark to finish, then validate/export once. No inference calls."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    status = root/'handoff_export_status.json'
    deadline = time.monotonic()+10800
    run.save(status, {'status': 'waiting_for_inference', 'updated_at_utc': run.now()})
    while time.monotonic() < deadline:
        m = run.read(root/'manifest.json')
        if m['status'] not in ['running', 'prepared']:
            if not (root/'source_validation.json').exists():
                time.sleep(10)
                continue
            run.save(status, {'status': 'validating_and_exporting', 'updated_at_utc': run.now()})
            with (root/'handoff_export.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, str(run.ROOT/'scripts/export_corrected_pdf_handoff.py'), '--root', str(root)], stdout=log, stderr=subprocess.STDOUT)
            run.save(status, {'status': 'completed' if result.returncode == 0 else 'incomplete_or_export_error',
                              'exit_code': result.returncode, 'updated_at_utc': run.now()})
            return result.returncode
        time.sleep(30)
    run.save(status, {'status': 'wait_timeout', 'updated_at_utc': run.now()})
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
