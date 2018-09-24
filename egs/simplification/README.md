## Text simplification

This trains on a complex and simplified dataset from <a href="https://newsela.com/">Newsela</a> of 94208 sentence pairs, drawn from the splits by Zhang and Lapata paper. The results should be comparable to their paper <a href="https://arxiv.org/pdf/1703.10931.pdf">Sentence Simplification with Deep Reinforcement Learning</a>.

In this example, we will run everything from a `$workdir` that is in egs/simplification source tree. We use the script "run.sh" and hyperparameter file "model1.hpm". 


Note that I've assumed that sockeye-recipes is in your home directory, so in "~/sockeye-recipes".

Now we will run the entire process through one script, assuming that you are on a GPU node. See the script for invocation and modify as needed. 
```bash
cd ~/sockeye-recipes/egs/simplification/
./run.sh
```

To change the model configuration then you need to modify "model1.hpm"
