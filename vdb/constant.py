import multiprocessing as mp
import os

HOME_PREFIX = '/home/moomou/dev/mlab/vdb'

CHECKPT_DIR = os.environ.get('CHK_DIR', os.path.join('./checkpoints'))
DATA_ROOT = '/home/moomou/dev/data'

CN_ROOT = os.path.join(DATA_ROOT, 'cn_wav')
DAN_ROOT = os.path.join(DATA_ROOT, 'danish')
TIMIT_ROOT = os.path.join(DATA_ROOT, 'timit', 'raw', 'TIMIT')
VCTK_ROOT = os.path.join(DATA_ROOT, 'vctk', 'data')
NOISE_ROOT = os.path.join(DATA_ROOT, 'bgNoise')
VOICE_ROOT = os.path.join(DATA_ROOT, 'voice')
FFF_EN_ROOT = os.path.join(DATA_ROOT, 'FE_FATE_EN')
FFH_JP_ROOT = os.path.join(DATA_ROOT, 'FE_HEROES_JAP')

TIMIT_SAMPLE = os.path.join(TIMIT_ROOT, 'TEST', 'DR1', 'FAKS0', 'SA1.WAV')
VCTK_SAMPLE = os.path.join(VCTK_ROOT, 'p225', 'p225_001.wav')
NOISE_SAMPLE = os.path.join(NOISE_ROOT, 'NPARK', 'ch05.wav')
FFF_EN_SAMPLE = os.path.join(FFF_EN_ROOT, 'Anankos/0x52743E0.WAV')
FFH_JP_SAMPLE = os.path.join(FFF_EN_ROOT, '01/VOICE_HECTOR_SKILL_3.mp3')

MFCC_SAMPLE_LEN_MS = 25
MFCC_NB_COEFFICIENTS = 32

SAMPLE_RATE = 8000 * 2
MAX_FREQ = SAMPLE_RATE / 2

CPU_COUNT = mp.cpu_count() // 4
