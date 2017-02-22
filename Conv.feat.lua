local Conv = torch.class('similarityMeasure.Conv')

function Conv:__init(config)
  self.mem_dim       = config.mem_dim       or 150 --200
  self.learning_rate = config.learning_rate or 0.001
  self.batch_size    = config.batch_size    or 1 --25
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 1e-5
  self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
  self.sim_nhidden   = config.sim_nhidden   or 150
  self.use_ext_feat  = config.use_ext_feat  or false
  self.task          = config.task          or 'twitter' --'twitter'  -- or 'vid'
  self.ext_feat_size = 0
  if self.use_ext_feat == true and self.task == 'trec' then
    self.ext_feat_size = 1
  elseif self.use_ext_feat == true and self.task == 'qa' then
    self.ext_feat_size = 4
  end
  print('use_ext_feat in Conv.lua: ' .. tostring(self.use_ext_feat) .. tostring(self.ext_feat_size))  
  print('call conv.feat.lua cos')
  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- number of similarity rating classes
  if self.task=='qa' or self.task == 'trec' or self.task == 'twitter' then
    self.num_classes = 2
    -- Negative Log Likelihood optimization objective
    self.criterion = nn.ClassNLLCriterion();
  elseif self.task == 'sic' then
    self.num_classes = 5
    self.criterion = nn.ClassNLLCriterion()
  else
    error("not possible task!")
  end	
  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  dofile 'models.lua'
  local modelName = 'SIGIR'
  self.ngram = 5
  self.NumFilter = 100
  self.nhid1 = 2*self.NumFilter+1+1
  self.length = self.emb_dim
  --self.convModel = createModel(modelName, 10000, self.length, self.num_classes, self.ngram, self.ext_feat_size)  
  self.convModel = self:paraConv()
  self.bilinearLayer = self:bilinear()
  self.model = nn.Sequential()
  self.model:add(self.convModel)
  self.model:add(self.bilinearLayer)
  self.linearLayer = self:linearLayer()
  local modules = nn.Parallel()
    :add(self.convModel)
    :add(self.bilinearLayer)
    :add(self.linearLayer) 
  ----------------------------------------
  self.params, self.grad_params = modules:getParameters()
end

function Conv:paraConv()
  local D = self.length
  local NumFilter = self.NumFilter
  local kW = self.ngram  

  local q_conv = nn.Sequential()
  q_conv:add(nn.TemporalConvolution(D, NumFilter, kW, 1))
  q_conv:add(nn.Tanh())
  q_conv:add(nn.Max(1))
  q_conv:add(nn.Reshape(1, NumFilter))
  local ans_conv = nn.Sequential()
  ans_conv:add(nn.TemporalConvolution(D, NumFilter, kW, 1))
  ans_conv:add(nn.Tanh())
  ans_conv:add(nn.Max(1))
  ans_conv:add(nn.Reshape(1, NumFilter))
  
  paraQuery=nn.ParallelTable()
  paraQuery:add(q_conv)
  paraQuery:add(ans_conv)
  return paraQuery
end

function Conv:bilinear()
  local NumFilter = self.NumFilter
  local linput, rinput = nn.Identity()(), nn.Identity()()
  local bi_dist = nn.Bilinear(NumFilter, NumFilter, 1, false){linput, rinput}
  --local cos_dist = nn.CosineDistance(){linput, rinput}
  --local dot_dist = nn.DotProduct(){linput, rinput} 
  local inputs = {linput, rinput}
  --local vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), dot_dist}
  local vec_feats = bi_dist
  --vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput)}
  return nn.gModule(inputs, {vec_feats}) 
end

function Conv:linearLayer()
  local toplayer = nn.Sequential()
  toplayer:add(nn.Linear(1, self.nhid1))
  toplayer:add(nn.Tanh())
  --toplayer:add(nn.Dropout(0.5))
  toplayer:add(nn.Linear(self.nhid1, self.num_classes))
  toplayer:add(nn.LogSoftMax())
  return toplayer
end

function Conv:trainCombineOnly(dataset)
  train_looss = 0.0
   
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  self.model:training()
  self.linearLayer:training()
  for i = 1, dataset.size, self.batch_size do
    --if i%10 == 1 then
    --	    xlua.progress(i, dataset.size)
    --end

    local batch_size = 1 
    -- get target distributions for batch
    local targets = torch.zeros(batch_size, self.num_classes)
    for j = 1, batch_size do
      local sim  = -0.1
      if self.task == 'sic' or self.task == 'vid' then
        sim = dataset.labels[indices[i + j - 1]] * (self.num_classes - 1) + 1
      elseif self.task == 'twitter' or self.task == 'trec' or self.task == 'qa' then
        sim = dataset.labels[indices[i + j - 1]] + 1 
      else
	error("not possible!")
      end      
    end
    
    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local sim = dataset.labels[idx] + 1 -- read class label
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local linputs = self.emb_vecs:index(1, lsent:long()):double()
        local rinputs = self.emb_vecs:index(1, rsent:long()):double()
        local output = nil
        --local conv_output = self.convModel:forward({linputs, rinputs})
        --print(conv_output[1], conv_output[2])
        --local bilinear_score = self.bilinearLayer:forward(conv_output)
        --local bilinear_feat = torch.cat({conv_output[1], conv_output[2], bilinear_score, dataset.ranks[idx]}, 2)
        local bilinear_output = self.model:forward({linputs, rinputs})
        --local bilinear_feat = torch.cat(bilinear_output, dataset.ranks[idx])
        local bilinear_feat = bilinear_output
        --print(bilinear_output:size(), bilinear_feat:size())
        local output = self.linearLayer:forward(bilinear_feat)
        local sim_grad = 0
        if self.task == 'vid' or self.task == 'sic' then
          loss = self.criterion:forward(output, targets[1])
          sim_grad = self.criterion:backward(output, targets[1])
        else
	  loss = self.criterion:forward(output, sim)
          sim_grad = self.criterion:backward(output, sim)
	end
        --print(dataset.ids[idx], dataset.doc_ids[idx], dataset.ranks[idx])
        --print(lsent, rsent)
        --print(output:exp(), sim, loss, sim_grad)
	--print(self.params:norm())
        train_looss = loss + train_looss
        local linear_feat_grad = self.linearLayer:backward(bilinear_feat, sim_grad)
        --local linear_grad = linear_feat_grad:narrow(1, 1, 1)
        --local linear_grad = linear_feat_grad:narrow(1, 1, 2*self.NumFilter+1)
        local linear_grad = linear_feat_grad
        --print(linear_feat_grad:size(), linear_grad:size())
        --local bilinear_grad = self.bilinearLayer:backward(conv_output, linear_feat_grad:narrow(2, 2*self.NumFilter+1, 1))
        --self.convModel:backward({linputs, rinputs}, linear_feat_grad:narrow(2, 1, 2*self.NumFilter))
        --self.convModel:backward({linputs, rinputs}, bilinear_grad)
        --print(linear_feat_grad, linear_grad)
        self.model:backward({linputs, rinputs}, linear_grad)
      end
      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end
    _, fs  = optim.sgd(feval, self.params, self.optim_state)
    --train_looss = train_looss + fs[#fs]
  end
  print('Loss: ' .. train_looss)
end

-- Predict the similarity of a sentence pair.
function Conv:predictCombination(lsent, rsent, ext_feat)
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()
  local bilinear_output = self.model:forward({linputs, rinputs})
  --local bilinear_feat = torch.cat(bilinear_output, ext_feat)
  local bilinear_feat = bilinear_output
  local output = self.linearLayer:forward(bilinear_feat)
  local val = -1.0
  if self.task == 'sic' then
    val = torch.range(1, 5, 1):dot(output:exp())
  elseif self.task == 'vid' then
    val = torch.range(0, 5, 1):dot(output:exp())
  elseif self.task == 'twitter' or self.task == 'trec' or self.task == 'qa' then
    return output:exp()[2]
  else
    error("not possible task")
  end
  return val
end

-- Produce similarity predictions for each sentence pair in the dataset.
function Conv:predict_dataset(dataset)
  self.model:evaluate()
  self.linearLayer:evaluate()
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    local lsent, rsent, feat = dataset.lsents[i], dataset.rsents[i], dataset.ranks[i]
    predictions[i] = self:predictCombination(lsent, rsent, feat)
    --print(dataset.ids[i], dataset.doc_ids[i])
    --print(lsent, rsent, feat)
    --print(dataset.labels[i], predictions[i])
  end
  return predictions
end

function Conv:print_config()
  local num_params = self.params:nElement()

  print('num params: ' .. num_params)
  print('word vector dim: ' .. self.emb_dim)
  print('regularization strength: ' .. self.reg)
  print('minibatch size: ' .. self.batch_size)
  print('learning rate: ' .. self.learning_rate)
  print('model structure: ' .. self.structure)
  --print('number of hidden layers: ' .. self.num_layers)
  --print('number of neurons in hidden layer: ' .. self.mem_dim)
end

