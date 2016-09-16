local Conv = torch.class('similarityMeasure.Conv')

function Conv:__init(config)
  self.mem_dim       = config.mem_dim       or 150 --200
  self.learning_rate = config.learning_rate or 0.001
  self.batch_size    = config.batch_size    or 1 --25
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 1e-5
  self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
  self.sim_nhidden   = config.sim_nhidden   or 150
  self.task          = config.task          or 'twitter' --'twitter'  -- or 'vid'
	
  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- number of similarity rating classes
  if self.task=='qa' then
    self.num_classes = 2
  else
    error("not possible task!")
  end
	
  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- Negative Log Likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion();
  dofile 'models.lua'
  local modelName = 'SIGIR'
  self.ngram = 5
  self.length = self.emb_dim
  self.convModel = createModel(modelName, 10000, self.length, self.num_classes, self.ngram)  
  
  ----------------------------------------
  self.params, self.grad_params = self.convModel:getParameters()
end


function Conv:trainCombineOnly(dataset)
  train_looss = 0.0
   
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  self.convModel:training()

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
      elseif self.task == 'twitter' or self.task == 'ttg' or self.task == 'qa' then
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
   	local output = self.convModel:forward({linputs, rinputs})
        local sim_grad = 0
        if self.task == 'vid' or self.task == 'sic' then
	  error("Not possible")
        else
	  loss = self.criterion:forward(output, sim)
          sim_grad = self.criterion:backward(output, sim)
	end
	train_looss = loss + train_looss
	self.convModel:backward({linputs, rinputs}, sim_grad)
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
function Conv:predictCombination(lsent, rsent)
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()

  local output = self.convModel:forward({linputs, rinputs})
  local val = -1.0
  if self.task == 'sic' then
    val = torch.range(1, 5, 1):dot(output:exp())
  elseif self.task == 'vid' then
    val = torch.range(0, 5, 1):dot(output:exp())
  elseif self.task == 'twitter' or self.task == 'ttg' or self.task == 'qa' then
    return output:exp()[2]
  else
    error("not possible task")
  end
  return val
end

-- Produce similarity predictions for each sentence pair in the dataset.
function Conv:predict_dataset(dataset)
  self.convModel:evaluate()
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predictCombination(lsent, rsent)
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

