# This is a test script that gets the first beam history and reconstructs the sentences.
import json
import sys
from collections import deque

if len(sys.argv) < 2:
    print('Usage: reconstruct_hyps.py [path to input]', file=sys.stderr)
    sys.exit(1)

with open(sys.argv[1], 'r') as beam_file:
    beam_lines = beam_file.readlines()

for line_num in range(len(beam_lines)):
    line = json.loads(beam_lines[line_num])
    tokens = line['predicted_tokens']
    parent_ids = line['parent_ids']
    BEAM_SIZE = len(tokens[0])
    sentence_tokens = [deque() for i in range(BEAM_SIZE)]

    for i in range(BEAM_SIZE):
        id = i
        # Reconstruct sentence from the tail of the beam
        for j in range(len(tokens)-1, -1, -1):
            token = tokens[j][id]
            if token != '</s>' and token != '<pad>':
                sentence_tokens[i].appendleft(token)
            id = parent_ids[j][id]

    sentence_tokens = [list(s) for s in sentence_tokens]
    print(f"Sentence {line_num+1}")
    for i, s in enumerate(sentence_tokens):
        print(f"{i+1}) {' '.join(s)}")
    #print(f"{[' '.join(s) for s in sentence_tokens]}")

    # n = 3
    # for sentence in sentence_tokens:
    #     print(list(zip(*[sentence[i:] for i in range(n)])))
    #     break
    if len(sys.argv) > 2:
        if line_num >= int(sys.argv[2])-1:
            break
    else:
        exit(0)
