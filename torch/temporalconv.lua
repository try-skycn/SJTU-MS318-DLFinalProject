require 'nn'
require 'optim'
require 'loader'

--[[Load Data & Preprocessing]]--
max_len = 128
ldr = Loader:new(nil, max_len)

function make_dataset(name)
    local dataset = {data = ldr.data[name], label = ldr.label[name]}
    function dataset:size() 
        return self.data:size(1) 
    end
    for i = 1, dataset:size() do
        dataset[i] = {dataset.data[i], dataset.label[i]}
    end
    for i = 1, 2 do
        -- we abandon engy...
        if i == 2 then
            dataset.data[{{}, {}, i}]:fill(0)
            break
        end
        local std = dataset.data[{{}, {}, i}]:std()
        dataset.data[{{}, {}, i}]:div(std)
        -- for j = 1, dataset:size() do
        --     local std = dataset.data[{j, {}, i}]:std()
        --     dataset.data[{j, {}, i}]:div(std)
        -- end
    end
    return dataset
end

trainset = make_dataset('train')
testset = make_dataset('test')
newtestset = make_dataset('test_new')

--[[Model]]--

local inp = 2;  -- dimensionality of one sequence element 
local outp = 4; -- number of derived features for one sequence element
local conv_kw = 4;   -- number of sequence element kernel operates on per step
local conv_dw = 4;   -- step dw and go on to the next sequence element
local pool_kw = 4;   -- number of sequence element pooled on per step
local pool_dw = 4;   -- step dw and go on to the next sequence element
local nOutputFrame = (max_len - conv_kw) / conv_dw + 1
local nOutputFrame = (nOutputFrame - pool_kw) / pool_dw + 1

net = nn.Sequential()
peek = nn.TemporalConvolution(inp, outp, conv_kw, conv_dw)
net:add(peek)
net:add(nn.ReLU())
net:add(nn.TemporalMaxPooling(pool_kw, pool_dw))
net:add(nn.View(outp*nOutputFrame))
net:add(nn.Linear(outp*nOutputFrame, 64))
-- net:add(nn.BatchNormalization(64))
net:add(nn.ReLU())
net:add(nn.Linear(64, 4))
net:add(nn.LogSoftMax())

print (net:__tostring())

--[[Train]]--

criterion = nn.ClassNLLCriterion();

trainer = nn.StochasticGradient(net, criterion)  
trainer.learningRate = 0.001
trainer.maxIteration = 200
trainer.verbose = False

function cal_acc(dataset)
    local acc = 0.0
    for i = 1, dataset:size() do
        _, output = torch.max(net:forward(dataset[i][1]), 1)
        if output[1] == dataset[i][2] then
            acc = acc + 1.0
        end
    end
    acc = acc / dataset:size()
    return acc
end

recorder = {loss = {}, trainacc = {}, valacc = {}, testacc = {}}

function tracker(obj, iteration, currentError) 
    local trainacc = cal_acc(trainset)
    local valacc = cal_acc(testset)
    local testacc = cal_acc(newtestset)
    table.insert(recorder.loss, currentError)
    table.insert(recorder.trainacc, trainacc)
    table.insert(recorder.valacc, valacc)
    table.insert(recorder.testacc, testacc)
    if (iteration % 10 == 0) then 
        print('Iteration '..iteration..': loss('..currentError..') train_acc('
            ..trainacc..') test_acc('..valacc..') new_test_acc('..testacc..').')
    end
end

trainer.hookIteration = tracker

trainer:train(trainset)
