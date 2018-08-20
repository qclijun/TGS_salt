import time
from contextlib import contextmanager


@contextmanager
def timer(title):
    start = time.time()
    yield
    print('{} - Done in {:.2f} secs'.format(title, time.time() - start))
