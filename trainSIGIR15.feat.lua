require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('Conv.feat.lua')
include('CsDis.lua')
include('metric.lua')
printf = utils.printf

-- global paths (modify if desired)
similarityMeasure.data_dir        = 'data'
similarityMeasure.models_dir      = 'trained_models'
similarityMeasure.predictions_dir = 'predictions'

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-dataset', 'TrecQA', 'dataset, can be TrecQA or WikiQA')
cmd:option('-version', 'raw', 'the version of TrecQA dataset, can be raw and clean')
cmd:option('-ext', false, 'whether use the external feature')
cmd:text()
opt = cmd:parse(arg)

--read default arguments
local args = {
  model = 'conv', --convolutional neural network 
  layers = 1, -- number of hidden layers in the fully-connected layer
  dim = 150, -- number of neurons in the hidden layer.
}

local model_name, model_class, model_structure
model_name = 'conv'
model_class = similarityMeasure.Conv
model_structure = model_name

--torch.seed()
torch.manualSeed(12345)
print('<torch> using the automatic seed: ' .. torch.initialSeed())
print('use external feature: ' .. tostring(opt.ext))
-- directory containing dataset files
--local data_dir = 'data/' .. opt.dataset .. '/'

if opt.dataset == 'Twitter' then
  data_dir = '/scratch1/jinfeng/DNN/data/' .. opt.dataset:lower() .. '/'
  vocab = similarityMeasure.Vocab(data_dir .. 'order_by_time/vocab.txt')
else
  data_dir = 'data/' .. opt.dataset .. '/'
  if opt.dataset == 'sick' then
    vocab = similarityMeasure.Vocab(data_dir .. 'vocab-cased.txt')
  else
    vocab = similarityMeasure.Vocab(data_dir .. 'vocab.txt')
  end
end
-- load vocab

-- load embeddings
local emb_dir = '/scratch1/jinfeng/DNN/data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local dim = '.300d'
if opt.dataset == 'Trec' then
  emb_dir = 'data/embedding/'
  emb_prefix = emb_dir .. 'twitter.word2vec'
  dim = '.50d'
end
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. dim .. '.th')

print('loading word embeddings' .. emb_prefix .. '.th')

local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
local unknown_vecs = {} 
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.25, 0.25)
    unknown_vecs[w] = vocab:freq(i)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
unknown_vecsfile = torch.DiskFile(data_dir .. 'unknown-vocab.txt', 'w')
for w, freq in pairs(unknown_vecs) do
  unknown_vecsfile:writeString(string.format('%s %s\n', w, freq))
end
unknown_vecsfile:close()
collectgarbage()

local taskD = 'qa'
-- load datasets
print('loading datasets ' .. opt.dataset)
if opt.dataset == 'TrecQA' then
  train_dir = data_dir .. 'train/'
  dev_dir = data_dir .. opt.version .. '-dev/'
  test_dir = data_dir .. opt.version .. '-test/'
elseif opt.dataset == 'WikiQA' or opt.dataset == 'sick' then
  train_dir = data_dir .. 'train/'
  dev_dir = data_dir .. 'dev/'
  test_dir = data_dir .. 'test/'
  if opt.dataset == 'sick' then 
    taskD = 'sic'
  end
elseif opt.dataset == 'Twitter' then
  train_dir = data_dir .. 'order_by_time/train_2011/'
  test_dir = data_dir .. 'order_by_time/test_2011/'
  taskD = 'twitter'
elseif opt.dataset == 'Trec' then
  train_dir = data_dir .. 'trec2011/neural-network/all/'
  test_dir = data_dir .. 'trec2012/neural-network/'
  all_tweets = similarityMeasure.load_tweets('data/Trec/all_top1000_tweets.txt')
  taskD = 'trec'
end

local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD, all_tweets, 'train')
printf('train_dir: %s, num train = %d\n', train_dir, train_dataset.size)

if opt.dataset == 'TrecQA' or opt.dataset == 'WikiQA' or opt.dataset == 'sick' or opt.dataset == 'Twitter' then
  if dev_dir ~= nil then
    dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD, nil, 'dev')
    printf('dev_dir: %s, num dev   = %d\n', dev_dir, dev_dataset.size)
  end
  test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD, nil, 'test')
  printf('test_dir: %s, num test  = %d\n', test_dir, test_dataset.size)
end

-- initialize model
local model = model_class{
  emb_vecs   = vecs,
  structure  = model_structure,
  num_layers = args.layers,
  mem_dim    = args.dim,
  task       = taskD,
  use_ext_feat = opt.ext,
}

-- number of epochs to train
local num_epochs = 25

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()


if lfs.attributes(similarityMeasure.predictions_dir) == nil then
  lfs.mkdir(similarityMeasure.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

-- threads
--torch.setnumthreads(4)
--print('<torch> number of threads in used: ' .. torch.getnumthreads())
header('Training model')

local id = 100000
print("Id: " .. id)

function test(test_dataset, epoch)
  local test_predictions = model:predict_dataset(test_dataset)
  if opt.dataset == 'sick' then
    test_map_score = pearson(test_predictions, test_dataset.labels)
    print('test pearson score:' .. test_map_score)
  else
    test_map_score = map(test_predictions, test_dataset.labels, test_dataset.boundary, test_dataset.numrels)
    local test_mrr_score = mrr(test_predictions, test_dataset.labels, test_dataset.boundary, test_dataset.numrels)
    local test_p30_score = p_30(test_predictions, test_dataset.labels, test_dataset.boundary)
    printf('-- test map score: %.4f, mrr score: %.4f, p30 score:%.4f\n', test_map_score, test_mrr_score, test_p30_score)
  end
  local predictions_save_path = string.format(
	similarityMeasure.predictions_dir .. '/results-%s.%dl.%dd.epoch-%d.%.5f.%d.pred', args.model, args.layers, args.dim, epoch, test_map_score, id)
  local predictions_file = torch.DiskFile(predictions_save_path, 'w')
  print('writing predictions to ' .. predictions_save_path)
  for i = 1, test_predictions:size(1) do
    predictions_file:writeString(string.format('%s Q0 %s %d %f %d\n', test_dataset.ids[i], test_dataset.doc_ids[i], i, test_predictions[i], test_dataset.labels[i]))
  end
  predictions_file:close()
end

for i = 1, num_epochs do
  local start = sys.clock()
  print('--------------- EPOCH ' .. i .. '--- -------------')
  model:trainCombineOnly(train_dataset)
  print('Finished epoch in ' .. ( sys.clock() - start) )
 
  if opt.dataset == 'TrecQA' or opt.dataset == 'WikiQA' or opt.dataset == 'Twitter' or opt.dataset == 'sick' then 
    if dev_dir ~= nil then
      local dev_predictions = model:predict_dataset(dev_dataset)
      if opt.dataset == 'sick' then
        local dev_score = pearson(dev_predictions, dev_dataset.labels)
        print('-- dev pearson score: ' .. dev_score)
      else
        local dev_map_score = map(dev_predictions, dev_dataset.labels, dev_dataset.boundary, dev_dataset.numrels)
        local dev_mrr_score = mrr(dev_predictions, dev_dataset.labels, dev_dataset.boundary, dev_dataset.numrels)
        printf('-- dev map score: %.5f, mrr score: %.5f\n', dev_map_score, dev_mrr_score)
      end
    end
    test(test_dataset, i)
  elseif opt.dataset == 'Trec' then
    for submit in lfs.dir(test_dir) do
      local submit_dir = test_dir .. submit .. '/'
      if submit ~= '.' and submit ~= '..' and submit ~= 'all' and lfs.attributes(submit_dir, "mode") == "directory" then
 	local test_dataset = similarityMeasure.read_relatedness_dataset(submit_dir, vocab, taskD, all_tweets, 'test')
        printf('test_dir: %s, num test  = %d\n', submit_dir, test_dataset.size)
        test(test_dataset, i)
      end
    end   
  end
 
end
