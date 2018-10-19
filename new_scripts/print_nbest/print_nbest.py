# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Modified by Alexander Rush, 2017

# MIT License
#
# Copyright (c) 2017-present The OpenNMT Authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Generate beam search visualization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json
import shutil
from string import Template

import networkx as nx
from networkx.readwrite import json_graph


# Pad token used in sockeye
# Used to filter out pad tokens from the graph
PAD_TOKEN = "<pad>"

def _add_graph_level(graph, level, parent_ids, names, scores, normalized_scores,
                     include_pad):
    """Adds a level to the passed graph"""
    for i, parent_id in enumerate(parent_ids):
        if not include_pad and names[i] == PAD_TOKEN:
            continue
        new_node = (level, i)
        parent_node = (level - 1, parent_id)
        raw_score = float(scores[i]) if scores[i] is not None else 100000.0
        norm_score =  float(normalized_scores[i]) if normalized_scores[i] is not None else 100000.0

        graph.add_node(new_node)
        graph.node[new_node]["name"] = names[i]
        graph.node[new_node]["score"] = raw_score
        graph.node[new_node]["norm_score"] = norm_score
        graph.node[new_node]["size"] = 100
        # Add an edge to the parent
        graph.add_edge(parent_node, new_node)

def create_graph(predicted_ids, parent_ids, scores, normalized_scores, include_pad):

    seq_length = len(predicted_ids)
    graph = nx.DiGraph()
    for level in range(seq_length):
        names = [pred for pred in predicted_ids[level]]
        _add_graph_level(graph, level + 1, parent_ids[level], names,
                         scores[level], normalized_scores[level], include_pad)
    graph.node[(0, 0)]["name"] = "START"
    return graph

## Extracts all complete sentences from the graph, and sorts them by normalized score
def extract_sentences(graph):
    sentences = []
    scores = []

    ## Extracts all complete sentences
    for n in graph.nodes():
        if graph.node[n]["name"] == "</s>":
            paths = nx.all_simple_paths(graph, source=(0,0), target=n)

            for path in paths:
                sentences.append([graph.node[node]["name"] for node in path[1:-1]])
                scores.append(graph.node[path[len(path)-1]]["norm_score"])

    ## Sorts sentences by normalized score
    sents_sorted = []
    while min(scores) < 1000000.0:
        ind = scores.index(min(scores))
        sents_sorted.append(sentences[ind])
        scores[ind] = 1000000.0              
    return sents_sorted
        
        

def generate(input_data, output_file, include_pad=False):
    sentences = []

    with open(input_data) as beams:
        for i, line in enumerate(beams):
            beam = json.loads(line)

            graph = create_graph(predicted_ids=beam["predicted_tokens"],
                                 parent_ids=beam["parent_ids"],
                                 scores=beam["scores"],
                                 normalized_scores=beam["normalized_scores"],
                                 include_pad=include_pad)

            sents = extract_sentences(graph)
            sentences.append(sents)

    with open(output_file, 'w', encoding='utf8') as f:
        for sents in sentences:
            f.write("\t".join([" ".join(s) for s in sents]) + "\n")

    

def main():
    parser = argparse.ArgumentParser(description="Generate nbest sentences")
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="path to the beam search data file")
    parser.add_argument(
        "-o", "--output_file", type=str, required=True,
        help="path to the output file")
    parser.add_argument('--pad', dest='include_pad', action='store_true')
    parser.add_argument('--no-pad', dest='include_pad', action='store_false')
    parser.set_defaults(include_pad=False)
    args = parser.parse_args()

    generate(args.data, args.output_file, include_pad=args.include_pad)


if __name__ == "__main__":
    main()
