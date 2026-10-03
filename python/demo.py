"""Offline text-generation demo, using the pinned downloaded model."""
from pathlib import Path
import sentencepiece as spm
from serving import Engine,load_model

root=Path(__file__).resolve().parents[1]
model=load_model(root/'models'/'stories260K.pt')
tokenizer=spm.SentencePieceProcessor(model_file=str(root/'models'/'tok512.model'))
engine=Engine(model)
try:
    prompt='Once upon a time, there was a little girl named Lily.'
    request=engine.submit('demo',[tokenizer.bos_id()]+tokenizer.encode(prompt),32)
    engine.run()
    if request.state!='completed':
        raise RuntimeError(request.error)
    print(tokenizer.decode(request.prompt+request.output))
finally:
    engine.close()
