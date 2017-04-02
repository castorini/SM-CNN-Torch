require('torch')
require('nn')
require('string')

similarityMeasure = {}
--include('util/utils.lua')
--include('util/read_session_data.lua')
include('../util/read_data.lua')
include('../util/Vocab.lua')
include('../build_embedding_feat.lua')
function save(dataset, path)
  local file = torch.DiskFile(path .. 'dfs_unigram_bigram_sim.txt', 'w')
  for i = 1, dataset.size do
    for j = 1, dataset.embedding_sim[i]:size(1) do
      file:writeString(string.format("%.3f ", dataset.embedding_sim[i][j]))
    end 
    file:writeString("\n") 
  end
  file:close()
end

print('loading word embeddings')
--local pretrain_data_dir = 'data/twitter/'
local taskD = 'twitter'
local qmax_len = 7 -- read the max length of all queries
print('max len of queries (check with your dataset!): ' .. qmax_len)
local emb_dir = '/mnt/research-6f/jinfeng/char-lstm/data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')

data_dir = 'data/twitter/'
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab_2011.txt')
local word2dfs = similarityMeasure.read_word2df(data_dir .. 'word2dfs_2011.txt', vocab)
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.25, 0.25)
  end
end
print('unk count = ' .. num_unk)

modes = {'train_2011/', 'dev_2011/', 'test_2011/'}
for i = 1, #modes do
  local dataset = similarityMeasure.read_relatedness_dataset(data_dir..modes[i], vocab, taskD)
  print('dir: %s, num = %d\n', modes[i], dataset.size)
  dataset.embedding_sim = {}
  for j = 1, dataset.size do
    local lsent, rsent = dataset.lsents[j], dataset.rsents[j]
    local linputs = vecs:index(1, lsent:long()):double()
    local rinputs = vecs:index(1, rsent:long()):double()
    dataset.embedding_sim[j] = comp_embedding_sim(lsent, linputs, rinputs, qmax_len, word2dfs)
  end
  save(dataset, data_dir .. modes[i])
end
