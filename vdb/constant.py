import os

DATA_ROOT = '/home/moomou/dev/data'

TIMIT_ROOT = os.path.join(DATA_ROOT, 'timit', 'raw', 'TIMIT')
VCTK_ROOT = os.path.join(DATA_ROOT, 'vctk', 'data')
CHECKPT_DIR = os.path.join('./checkpoints')

TIMIT_SAMPLE = os.path.join(TIMIT_ROOT, 'TEST', 'DR1', 'FAKS0', 'SA1.WAV')
VCTK_SAMPLE = os.path.join(VCTK_ROOT, 'p225', 'p225_001.wav')

MFCC_SAMPLE_LEN_MS = 25
MFCC_NB_COEFFICIENTS = 26

SAMPLE_RATE = 8000 * 2
MAX_FREQ = SAMPLE_RATE / 2
