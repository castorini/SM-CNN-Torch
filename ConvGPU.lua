local Conv = torch.class('similarityMeasure.Conv')

function Conv:__init(config)
  self.mem_dim       = config.mem_dim       or 150 --200
  self.learning_rate = config.learning_rate or 0.01
  self.batch_size    = config.batch_size    or 1
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
  self.criterion = nn.ClassNLLCriterion():cuda();
  dofile 'models.sigir15GPU.lua'
  local modelName = 'SIGIR'
  self.ngram = 3
  self.length = self.emb_dim
  self.convModel = createModel(modelName, 10000, self.length, self.num_classes, self.ngram)
  ----------------------------------------
  self.params, self.grad_params = self.convModel:getParameters()
  print(self.convModel:parameters()[1]:norm(), self.grad_params:norm())
  print(self.convModel:parameters()[1]:norm(), self.params:norm())
end


function Conv:trainCombineOnly(dataset)
  local train_looss = 0.0

  self.convModel:cuda()
  self.criterion:cuda()
  local indices = torch.randperm(dataset.size)
  self.convModel:training()
  local targets = torch.zeros(dataset.size, self.num_classes):cuda()

    -- get target distributions for batch
    for j = 1, dataset.size do
        local sim  = -0.1
        if self.task == 'sic' or self.task == 'vid' then
            sim = dataset.labels[indices[j]] * (self.num_classes - 1) + 1
        elseif self.task == 'qa' then
            sim = dataset.labels[indices[j]] + 1
        else
        error("not possible!")
        end
        local ceil, floor = math.ceil(sim), math.floor(sim)
        if ceil == floor then
            targets[{j, floor}] = 1
        else
            targets[{j, floor}] = ceil - sim
            targets[{j, ceil}] = sim - floor
        end--]]
    end

  local num_batch = math.ceil(dataset.size*1.0/self.batch_size)
  for i = 1, num_batch do
    if i%10 == 1 then
        xlua.progress(i, num_batch)
    end

    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      local batch_size = self.batch_size
      if i == num_batch then
        batch_size = dataset.size - (i-1)*self.batch_size
      end
      local linputs = torch.zeros(batch_size, dataset.lmax, self.emb_dim):cuda()
      local rinputs = torch.zeros(batch_size, dataset.rmax, self.emb_dim):cuda()
      --print(linputs:size())
      local batch_targets = torch.zeros(batch_size):cuda()
      for j = 1, batch_size do
        local idx = indices[(i-1)*self.batch_size + j]
        local sim = dataset.labels[idx] + 1 -- read class label
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        linputs[j] = self.emb_vecs:index(1, lsent:long())
        rinputs[j] = self.emb_vecs:index(1, rsent:long())
        batch_targets[j] = sim --targets[idx]
      end
      local output = self.convModel:forward({linputs, rinputs})
      loss = self.criterion:forward(output, batch_targets)
      sim_grad = self.criterion:backward(output, batch_targets)
      train_looss = loss + train_looss
      self.convModel:backward({linputs, rinputs}, sim_grad)
      print(i, loss, self.params:norm(), self.grad_params:norm())
      print(output, batch_targets, sim_grad, self.convModel:parameters()[1]:norm())
      -- regularization
      --loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      --self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end
    _, fs  = optim.sgd(feval, self.params, self.optim_state)
    --train_looss = train_looss + fs[#fs]
  end
  print('Loss: ' .. train_looss)
end

-- Predict the similarity of a sentence pair.
function Conv:predictCombination(linputs, rinputs)
  local output = self.convModel:forward({linputs, rinputs})
  local val = -1.0
  if self.task == 'sic' then
    val = torch.range(1, 5, 1):dot(output:exp())
  elseif self.task == 'vid' then
    val = torch.range(0, 5, 1):dot(output:exp())
  elseif self.task == 'twitter' or self.task == 'ttg' or self.task == 'qa' then
    return output:exp():select(2, 1)
  else
    error("not possible task")
  end
  return val
end

-- Produce similarity predictions for each sentence pair in the dataset.
function Conv:predict_dataset(dataset)
  self.convModel:cuda()
  self.convModel:evaluate()
  local predictions = torch.Tensor(dataset.size)
  local num_batch = math.ceil(dataset.size*1.0/self.batch_size)
  for i = 1, num_batch do
    if i%10 == 1 then
        xlua.progress(i, num_batch)
    end
    local batch_size = self.batch_size
    if i == num_batch then
      batch_size = dataset.size - (i-1)*self.batch_size
    end
    local linputs = torch.zeros(batch_size, dataset.lmax, self.emb_dim):cuda()
    local rinputs = torch.zeros(batch_size, dataset.rmax, self.emb_dim):cuda()
    for j = 1, batch_size do
      local idx = (i-1)*self.batch_size + j
      local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
      linputs[j] = self.emb_vecs:index(1, lsent:long())
      rinputs[j] = self.emb_vecs:index(1, rsent:long())
    end
    local output = self.convModel:forward({linputs, rinputs})
    local bindex, eindex = (i-1)*self.batch_size+1, (i-1)*self.batch_size+batch_size
    predictions:sub(bindex, eindex):copy(output:exp():select(2, 1))
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
