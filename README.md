# Reproduction
Reproduction of SIGIR15

Can keep updating/using this repo so it is easier to check progress. 

Run linear model with external features on twitter dataset:

$ ``th trainSIGIR15.lua -dataset twitter -model linear -ext``

To get the baseline numbers from original rankings:

$ ``th testMap.lua``

# Dataset 
The original dataset follows certain format:

- a.toks (or tokenize_query2.txt): sentence A, each sentence per line.
- b.toks (or tokenize_doc2.txt): sentence B, each sentence per line.
- id.txt: sentence pair ID
- sim.txt: ground truth label, the set of labels will be {0, 1}.
- numrels.txt: each line is the number of relevant docs for a query
- boundary.txt: start and end position of docs belonging to a query
