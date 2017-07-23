import os

DATA_ROOT = '/home/moomou/dev/data'

TIMIT_ROOT = os.path.join(DATA_ROOT, 'timit', 'raw', 'TIMIT')
VCTK_ROOT = os.path.join(DATA_ROOT, 'vctk', 'data')
CHECKPT_DIR = os.path.join('./checkpoints')

SAMPLE_RATE = 8000 * 2
MAX_FREQ = SAMPLE_RATE / 2
