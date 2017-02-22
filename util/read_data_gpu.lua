require('cutorch')

function similarityMeasure.read_embedding(vocab_path, emb_path)
  local vocab = similarityMeasure.Vocab(vocab_path)
  print(emb_path)
  local embedding = torch.load(emb_path):cuda()
  return vocab, embedding
end

function similarityMeasure.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  local max_len = 0

  local fixed = true
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    if max_len < len then max_len = len end
    local padLen = len
    if fixed and len < 3 then
      padLen = 3
    end
    local sent = torch.IntTensor(padLen):cuda()
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    if fixed and len < 3 then
      for i = len+1, padLen do
        sent[i] = vocab:index("<unk>") -- sent[len]
      end
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences, max_len
end

function similarityMeasure.padding(dataset, vocab)
  print('padding', dataset.lmax, dataset.rmax)
  for i = 1, dataset.size do
    local pad_lsent = torch.IntTensor(dataset.lmax):cuda()
    local pad_rsent = torch.IntTensor(dataset.rmax):cuda()
    for j = 1, dataset.lmax do
      if j <= dataset.lsents[i]:nElement() then
        pad_lsent[j] = dataset.lsents[i][j]
      else
        pad_lsent[j] = vocab:index("<unk>")
      end
    end
    for j = 1, dataset.rmax do
      if j <= dataset.rsents[i]:nElement() then
        pad_rsent[j] = dataset.rsents[i][j]
      else
        pad_rsent[j] = vocab:index("<unk>")
      end
    end

    dataset.lsents[i], dataset.rsents[i] = pad_lsent, pad_rsent
  end


end

function similarityMeasure.read_relatedness_dataset(dir, vocab, task)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsents, dataset.lmax = similarityMeasure.read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents, dataset.rmax = similarityMeasure.read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.lsents

  if task == 'twitter' or task == 'qa' then
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
  print(dir .. 'id.txt')
  local id_file = io.open(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = {}
--  dataset.labels = torch.Tensor(dataset.size)
  dataset.labels = torch.CudaTensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:read()
    if task == 'sic' then
        dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
    elseif task == 'vid' then
        dataset.labels[i] = 0.2 * (sim_file:readDouble()) -- vid data
    else
        dataset.labels[i] = (sim_file:readDouble()) -- twi and msp
    end
  end
  id_file:close()
  sim_file:close()
  return dataset
end
