import numpy as np
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import time
import os
import torch
from transformers import GPT2Tokenizer, GPT2Model

ckpt_name = sys.argv[1]
ckpt = torch.load(ckpt_name)[1]
matrix = ckpt["projector1"][0].matmul(ckpt["projector2"][0].transpose(1, 0)).cpu()
S, V, D = torch.linalg.svd(matrix)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')
embeddings=model.wte.weight

API_KEY = os.getenv("GOOGLE_API_KEY")
assert API_KEY != "none", "Please set the API_KEY before proceeding"
client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

responses = {f'gen-{i}-{j}': None for i in range(10) for j in range(2)}
not_done = np.ones(20) * 0.05
def response_callback(request_id, response, exception):
    responses[request_id] = (response, exception)
    if exception is None:
        not_done[int(request_id.split("-")[-1])] = 0
    if exception is not None:
        not_done[int(request_id.split("-")[-1])] = 1
        print(request_id, exception)

batch_request = client.new_batch_http_request()
for _i in range(10):
    print(_i)
    side_tokens = embeddings.matmul(D[_i]).argsort()[-20:].flip(0)
    print(tokenizer.convert_ids_to_tokens(side_tokens))
    text = tokenizer.decode(side_tokens)
    analyze_request= {
        'comment': {'text': text},
        'requestedAttributes': {"TOXICITY":{}},
        'spanAnnotations': True,
        "languages": ["en"],
    }
    batch_request.add(client.comments().analyze(body=analyze_request), callback=response_callback, request_id=f"gen-{_i}-0")

    side_tokens = embeddings.matmul(D[_i]).argsort()[:20]
    print(tokenizer.convert_ids_to_tokens(side_tokens))
    text = tokenizer.decode(side_tokens)
    analyze_request= {
        'comment': {'text': text},
        'requestedAttributes': {"TOXICITY":{}},
        'spanAnnotations': True,
        "languages": ["en"],
    }
    batch_request.add(client.comments().analyze(body=analyze_request), callback=response_callback, request_id=f"gen-{_i}-1")

batch_request.execute()
for key, _response in responses.items():
    print(key, _response[0]["detectedLanguages"], _response[0]["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
