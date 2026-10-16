import json
import subprocess
import sys
from pathlib import Path


def test_offline_demo(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root/'demo.py'), '--steps', '3', '--noise-samples', '2',
               '--grid', '4', '--output', str(tmp_path), '--device', 'cpu']
    subprocess.run(command, check=True, cwd=root, capture_output=True, text=True, timeout=120)
    for name in ('demo_output.png','layerwise.png','attribution.npz','summary.json'):
        assert (tmp_path/name).stat().st_size > 0
    data = json.loads((tmp_path/'summary.json').read_text())
    assert list(data['layerwise_cid_nats']) == ['Shallow','Mid','Output']
