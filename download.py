"""Download the pinned official dataset. A partial transfer is never accepted."""
import hashlib
from pathlib import Path
import urllib.request

REVISION = 'dd6f3a19eef5866e346c3270e098baa641a44948'
EXPECTED_SHA256 = 'b92f6c97277a554c8a7d6cfea900eca51639fd53a3e6c4fa58b82e35fd35804d'


def verify(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != EXPECTED_SHA256:
        raise ValueError('Dataset checksum mismatch. Do not train on this file.')


def main():
    target = Path('data/raw/multi_event.parquet')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify(target)
        print('Existing real dataset passed the pinned SHA256 check.')
        return
    url = f'https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}/flat/50m/multi_event.parquet'
    temporary = target.with_suffix('.partial')
    print('Downloading official Yambda-50M multi-event file...', flush=True)
    urllib.request.urlretrieve(url, temporary)
    import pyarrow.parquet as pq
    metadata = pq.ParquetFile(temporary).metadata
    if metadata.num_rows < 40_000_000:
        raise ValueError('Unexpected dataset size; refusing to continue.')
    verify(temporary)
    temporary.rename(target)
    print(f'Downloaded {metadata.num_rows:,} real events to {target}')


if __name__ == '__main__':
    main()
