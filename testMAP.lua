require('torch')
require('sys')

similarityMeasure = {}
include('util/Vocab.lua')
include('util/read_data.lua')
include('metric.lua')
printf = utils.printf

local data_dir = 'data/twitter/'
local test_dir = data_dir .. 'test_2011/'
local train_dir = data_dir .. 'train_2011/'
local dev_dir = data_dir .. 'dev_2011/'
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab.txt')
print('train dir:' .. train_dir)
print('test dir:' .. test_dir)
print('dev dir:' .. dev_dir)

local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, 'twitter')
local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, 'twitter')
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, 'twitter')

--local prediction_file, _ = io.open(test_dir .. 'pointwise-predictions.txt')
--local prediction, counter = torch.Tensor(test_dataset.size), 0
--while true do
--  line = prediction_file:read()
--  if line == nil then break end
--  counter = counter + 1
--  prediction[counter] = tonumber(line)
--end
--prediction_file:close()
local i = test_dataset.size
local rank = torch.Tensor(test_dataset.size):apply(function() i = i-1 return i end)

local test_map_score = map(rank, test_dataset.labels, test_dataset.boundary, test_dataset.numrels)

local test_mrr_score = mrr(rank, test_dataset.labels, test_dataset.boundary, test_dataset.numrels)

local test_p30_score = p_rank(rank, test_dataset.labels, test_dataset.boundary, 30)
local test_p5_score = p_rank(rank, test_dataset.labels, test_dataset.boundary, 5)
local test_p15_score = p_rank(rank, test_dataset.labels, test_dataset.boundary, 15)
local test_p100_score = p_rank(rank, test_dataset.labels, test_dataset.boundary, 100)

printf('[Testing]-- map score: %.4f, p30 score: %.4f, p5 score:%.4f, p15 score: %.4f, p100 score: %.4f, mrr score: %.4f\n', test_map_score, test_p30_score, test_p5_score, test_p15_score, test_p100_score, test_mrr_score)

local i = train_dataset.size
local rank = torch.Tensor(train_dataset.size):apply(function() i = i-1 return i end)

local train_map_score = map(rank, train_dataset.labels, train_dataset.boundary, train_dataset.numrels)

local train_p30_score = p_rank(rank, train_dataset.labels, train_dataset.boundary, 30)
printf('[Training]-- map score: %.4f, p30 score: %.4f\n', train_map_score, train_p30_score)

local i = dev_dataset.size
local rank = torch.Tensor(dev_dataset.size):apply(function() i = i-1 return i end)

local dev_map_score = map(rank, dev_dataset.labels, dev_dataset.boundary, dev_dataset.numrels)

local dev_p30_score = p_rank(rank, dev_dataset.labels, dev_dataset.boundary, 30)
printf('[DEV]-- map score: %.4f, p30 score: %.4f\n', dev_map_score, dev_p30_score)


