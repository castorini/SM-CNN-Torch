--[[

  Functions for loading data from disk.

--]]

function similarityMeasure.read_embedding(vocab_path, emb_path)
  local vocab = similarityMeasure.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function similarityMeasure.sent2embedding(sent, vocab)
  local tokens = stringx.split(sent)
  local len = #tokens
  local sentVec = torch.IntTensor(math.max(len,5))
  local counter = 0
  for i = 1, len do
    local token = tokens[i]
    sentVec[i] = vocab:index(token)
  end
  if len < 5 then
    for i = len+1, 5 do
      sentVec[i] = vocab:index('unk') -- sent[len]
    end
  end
  return sentVec
end

function similarityMeasure.read_sentences(path, vocab, all_tweets)
  local sentences = {}
  local file = io.open(path, 'r')
  local len_tweet_id = 17
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local sent = nil
    if #stringx.split(line) == 1 and tonumber(line) ~= nil and #line == len_tweet_id then  
      sent = all_tweets[line]
      if sent == nil then print(line) end
    else
      sent = line
    end
    --print(line, string.len(line))
    --print(sent)
    local sentVec = similarityMeasure.sent2embedding(sent, vocab)
    if sentVec == nil then print('line: '..line) end
    sentences[#sentences + 1] = sentVec
  end

  file:close()
  return sentences
end

function similarityMeasure.read_relatedness_dataset(dir, vocab, task, all_tweets, train_or_test)
  local dataset = {}
  dataset.vocab = vocab
  if task == 'trec' then
    file1 = 'query.txt'
    file2 = 'doc.txt'
  elseif task == 'twitter' then
    file1 = 'tokenize_query2.txt'
    file2 = 'tokenize_doc2.txt'
  else 
    file1 = 'a.toks'
    file2 = 'b.toks'
  end
  dataset.lsents = similarityMeasure.read_sentences(dir .. file1, vocab, all_tweets)
  dataset.rsents = similarityMeasure.read_sentences(dir .. file2, vocab, all_tweets)
  dataset.size = #dataset.lsents
  local id_file = io.open(dir .. 'id.txt', 'r')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = {}
  dataset.doc_ids = {}
  dataset.ranks = {}
  dataset.labels = torch.Tensor(dataset.size)
  if task == 'twitter' or task == 'qa' or task == 'trec' then  
    local boundary_file, _ = io.open(dir .. 'boundary.txt')
    local numrels_file = torch.DiskFile(dir .. 'numrels.txt')
    local boundary, counter = {}, 0
    while true do
      line = boundary_file:read()
      if line == nil then break end
      counter = counter + 1
      boundary[counter] = tonumber(line)
    end
    boundary_file:close()  
    dataset.boundary = torch.IntTensor(#boundary)
    for counter, bound in pairs(boundary) do
      dataset.boundary[counter] = bound
    end  
    -- read numrels data
    dataset.numrels = torch.IntTensor(#boundary-1)
    for i = 1, #boundary-1 do
      dataset.numrels[i] = numrels_file:readInt()
    end
    numrels_file:close()
  end

  for i = 1, dataset.size do
    dataset.ids[i] = id_file:read()
    if task == 'twitter' or task == 'trec' or task == 'qa' then
      dataset.labels[i] = (sim_file:readDouble()) -- twi and msp
    elseif task == 'sic' then
    	dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
    elseif task == 'vid' then
	dataset.labels[i] = 0.2 * (sim_file:readDouble()) -- vid data
    end
    if task == 'trec' then
      rank = stringx.split(dataset.ids[i], " ")
      dataset.ids[i], dataset.doc_ids[i] = rank[1], rank[3]
      avgNormRank, normRank = rank[4], rank[5]
      if train_or_test == 'train' then
        dataset.ranks[i] = torch.Tensor(1):fill(tonumber(avgNormRank))
      elseif train_or_test == 'test' then
        dataset.ranks[i] = torch.Tensor(1):fill(tonumber(normRank))
      end
    end
  end
  if task == 'qa' then
    f_wordlap = io.open(dir .. 'word_overlap.txt', 'r')
    for i = 1, dataset.size do
      wordlap_feat = stringx.split(f_wordlap:read(), " ")
      dataset.ranks[i] = torch.Tensor(4)
      for j = 1, 4 do  
        dataset.ranks[i][j] = tonumber(wordlap_feat[j])
      end
    end
  end

  id_file:close()
  sim_file:close()
  return dataset
end

function similarityMeasure.len(table)
  local count = 0
  for _ in pairs(table) do count = count + 1 end
  return count
end

function similarityMeasure.load_tweets(path)
  print('loading' .. path)
  local tweets_file = io.open(path, 'r')
  local all_tweets = {}
  local count, emptyCounter = 0, 0
  while true do
    line = tweets_file:read()
    if line == nil or string.len(line) == 0 then 
      break
    end
    tweetId = stringx.split(line, " ")[1]
    if string.len(line) == string.len(tweetId) then
      --print(tweetId, count)
      all_tweets[tweetId] = ''
      emptyCounter = emptyCounter + 1
    else
      all_tweets[tweetId] = string.sub(line, string.len(tweetId)+1)
    end
    count = count + 1
  end
  print(count, emptyCounter, similarityMeasure.len(all_tweets))
  return all_tweets
end
