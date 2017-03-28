require('torch')
require('nn')

function comp_embedding_sim(qwords, qvec, dvec, qmax, word2dfs)
  -- qwords: a list of word index for query
  -- qvec: a n*d tensor for each query, n is the number of words in query, d is the dimension of word vector
  -- dvec: a m*d tensor for each doc, m is the number of words in doc
  -- qmax: the maximum length of query in the collection
  -- word2dfs: a dict mapping each word index to a IDF value.
  local qlen, dlen = qvec:size(1), dvec:size(1)
  local unigram_sims = torch.zeros(qmax, dlen)
  local bigram_sims = torch.zeros(qmax-1, dlen-1)
  local unigram_dfs = torch.ones(qmax)*500
  local bigram_dfs = torch.ones(qmax-1)*500
  local cosine = nn.CosineDistance()
  for i = 1, qlen do
    unigram_dfs[i] = word2dfs[qwords[i]]
    for j = 1, dlen do
      unigram_sims[i][j] = cosine:forward({qvec[i], dvec[j]})[1]
      if i < qlen and j < dlen then
        q_bigram = (qvec[i]+qvec[i+1])/2
        d_bigram = (dvec[j]+dvec[j+1])/2
        bigram_sims[i][j] = cosine:forward({q_bigram, d_bigram})[1]
        bigram_dfs[i] = word2dfs[qwords[i]] + word2dfs[qwords[i+1]]
      end
    end
  end
  
  --print(unigram_dfs, bigram_dfs)
  local sort_uscore, sort_uindex = torch.sort(unigram_dfs, false) -- sort unigram doc frequency from low to high  
  local sort_bscore, sort_bindex = torch.sort(bigram_dfs, false) -- sort bigram doc freq from low to high
  --print(sort_uscore, sort_uindex)

  local unigram_max = torch.max(unigram_sims, 2)
  local bigram_max = torch.max(bigram_sims, 2)
  local sort_unigram_max = torch.zeros(qmax)
  local dfs_unigram_max = torch.zeros(qmax)
  local sort_bigram_max = torch.zeros(qmax-1)
  local dfs_bigram_max = torch.zeros(qmax-1)  
  for i = 1, qlen do
    sort_unigram_max[i] = unigram_max[sort_uindex[i]]
    dfs_unigram_max[i] = unigram_max[sort_uindex[i]] / (math.exp(sort_uscore[i]) + 1)
    if i < qlen then
      sort_bigram_max[i] = bigram_max[sort_bindex[i]]
      dfs_bigram_max[i] = bigram_max[sort_bindex[i]] / (math.exp(sort_bscore[i]) + 1)
    end
    --print(1/ (math.exp(sort_uscore[i]) + 1), sort_unigram_max[i], dfs_unigram_max[i])
    --print(1/ (math.exp(sort_bscore[i]) + 1), sort_bigram_max[i], dfs_bigram_max[i])
  end
  --print(sort_unigram_max, unigram_max)
  --print(sort_bigram_max, sort_unigram_max)
  return torch.cat({sort_unigram_max, sort_bigram_max, dfs_unigram_max, dfs_bigram_max}, 1)
end


