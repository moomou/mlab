import os
import multiprocessing as mp

THREAD_POOL = os.environ.get('TP', 'SPK')
POOL_SIZE = int(os.environ.get('PS', mp.cpu_count() // 2))
