from pathlib import Path
import sys

core_path = str(Path(__file__).resolve().parent / 'core')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

def preload(parser):
    parser.add_argument('--encrypt-pass', type=str, help='The password to enable image encryption.', default=None)