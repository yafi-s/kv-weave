"""Explicit, bounded download of two pinned MIT-licensed public model files."""
import hashlib
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REVISION='0bd21da7698eaf29a0d7de3992de8a46ef624add'
FILES={
    'stories260K.pt':'eec953f9d0f139e894ef8996302680e64b24813c7a98425424f5c85f7cf4abb1',
    'tok512.model':'dfff07d929db979913f166ec94a6f5ecad4c70cfed8eb5c9cbe7e464455e46f5',
}


def main():
    destination=ROOT/'models'
    destination.mkdir(exist_ok=True)
    for name,digest in FILES.items():
        path=destination/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
            continue
        url=f'https://huggingface.co/karpathy/tinyllamas/resolve/{REVISION}/stories260K/{name}'
        with urllib.request.urlopen(url,timeout=30) as response:
            data=response.read(2_000_001)
        if len(data)>2_000_000 or hashlib.sha256(data).hexdigest()!=digest:
            raise RuntimeError('model size or checksum mismatch')
        path.write_bytes(data)
    (destination/'REVISION').write_text(REVISION+'\n',encoding='utf8')
    print('Pinned model and tokenizer verified; total download about 1 MB.')


if __name__=='__main__':
    main()
