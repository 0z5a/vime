"""Verify unchanged owned inputs and stage four controls on the continued H20."""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK = Path('/root/autodl-tmp/0z5a/flashrlt-h20-20261006')
PREP = TASK / 'learner-prep-v1'
LOCKS = Path('/root/autodl-tmp/0z5a-coordination/h20')
UUID = 'GPU-8ee84e7d-143f-dd29-1097-85943783e027'


def main() -> None:
    request = json.load(sys.stdin)
    with (LOCKS / 'gpu0.lock').open('rb') as gpu, (LOCKS / 'io.lock').open('rb') as disk:
        assert [(os.fstat(x.fileno()).st_dev, os.fstat(x.fileno()).st_ino) for x in (gpu, disk)] == [(64528, 12920245107), (64528, 12920245108)]
        for handle in (gpu, disk):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '797c630e-4da1-48bd-bdf3-daabf26649d2'
        assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == UUID
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        assert subprocess.check_output(['stat', '-f', '-c', '%T', str(TASK)], text=True).strip() == 'xfs'
        free = shutil.disk_usage(TASK).free
        assert free >= 84 * 1024**3
        sources = json.loads((PREP / 'native-sources.json').read_text())
        assert sources['sources']['vime']['commit'] == 'd6229d638b08a3927d23d39854aa13a85df71933'
        for row in sources['files']:
            path = PREP / 'sources' / row['source'] / row['path']
            assert path.stat().st_dev == 64528
            data = path.read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        model = json.loads((PREP / 'model-manifest.json').read_text())
        assert model['revision'] == '3aaa2224253a92ca45cf2e3d427c360e1ef9c93d'
        for row in model['files']:
            path = TASK / 'model-stage-v2/models' / row['path']
            assert path.stat().st_dev == 64528 and path.stat().st_size == row['bytes']
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        packages = json.loads(subprocess.check_output([str(TASK.parent / 'envs/flashrlt-cu130/bin/python'), '-m', 'pip', 'list', '--format=json'], text=True, env=env))
        control = TASK / 'epoch2-control'
        control.mkdir()
        for row in request['files']:
            data = row['text'].encode()
            assert hashlib.sha256(data).hexdigest() == row['sha256']
            path = PREP / 'mcore-controller-epoch2.sh' if row['name'] == 'h20_mcore_gate_epoch2.sh' else control / row['name']
            with path.open('xb') as target:
                target.write(data)
        receipt = {'state': 'CURRENT_DEVICE_OWNED_SOURCE_MODEL_SPACE_AND_CONTROL_STAGE_PASS', 'pid': os.getpid(), 'gpu_uuid': UUID, 'free_bytes': free, 'source_files_verified': len(sources['files']), 'model_files_verified': len(model['files']), 'model_revision': model['revision'], 'controls': {row['name']: row['sha256'] for row in request['files']}, 'packages': packages, 'GPU_children': 0, 'environment_updated': False}
        (control / 'io-preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
