bazel-bin/lm_1b/lm_1b_eval --mode eval \
--pbtxt data/graph-2016-09-10.pbtxt \
--vocab_file data/vocab-2016-09-10.txt  \
--input_data data/test.txt \
--ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval --mode predict_perp \
--pbtxt data/graph-2016-09-10.pbtxt \
--vocab_file data/vocab-2016-09-10.txt  \
--input_data ~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.400beam.20centroids \
--output_data ~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.400beam.top1reranked_perplexity \
--ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval --mode predict_perp \
--pbtxt data/graph-2016-09-10.pbtxt \
--vocab_file data/vocab-2016-09-10.txt  \
--input_data data/tokens.TEST \
--output_data output/tokens.400beam.top1reranked_perplexity \
--ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval --mode predict_perp \
--pbtxt data/graph-2016-09-10.pbtxt \
--vocab_file data/vocab-2016-09-10.txt  \
--input_data data/tokens.400beam.20centroids \
--output_data output/tokens.400beam.top1reranked_perplexity \
--ckpt 'data/ckpt-*'