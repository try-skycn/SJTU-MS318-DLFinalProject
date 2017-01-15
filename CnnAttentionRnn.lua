require 'dp'
require 'rnn'
require 'ToneSource'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Temporal Convolutional Network for Tone Classification')
cmd:text('Example:')
cmd:text('$> th TemporalCnn.lua --learningRate 0.01 <')
cmd:text('Options:')
cmd:option('--load_mode', 'linear', 'linear | quad | shift')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--minLR', 0.0001, 'minimum learning rate')
cmd:option('--saturateEpoch', 100, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 8, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 300, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ convolutional layer ]]--
cmd:option('--inputC', 2, "dimensionality of one sequence element")
cmd:option('--outputC', 4, "number of derived features for one sequence element")
cmd:option('--convkw', 4, 'number of sequence element kernel operates on per step')
cmd:option('--convdw', 4, 'step dw and go on to the next sequence element')
cmd:option('--poolkw', 4, 'number of sequence element pooled on per step when pooling')
cmd:option('--pooldw', 4, 'step dw and go on to the next sequence element when pooling')

--[[ recurrent layer ]]--
cmd:option('--rnnHiddenSize', {64}, "hidden size of recurrent layer")
cmd:option('--lstm', false, 'use long short term memory')

--[[ fully connected layer ]]--
cmd:option('--dropout', 0, 'apply dropout with this probability')
-- cmd:option('--hiddenSize', 64, 'umber of hidden units used in FC')

--[[ data ]]--
cmd:option('--dataset', 'Tone', 'dataset name: Tone')
cmd:option('--maxLen', 64, 'length of input data')
cmd:option('--trainEpochSize', 800, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--noTest', false, 'dont propagate through the test set')

--[[ log & serialization ]]--
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--logDir', './data/log', 'saves training log of experiment')
cmd:option('--saveDir', './data/save', 'path for saving version of the subject with the lowest error')
cmd:option('--overwrite', false, 'overwrite checkpoint')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
	table.print(opt)
end

--[[Data Loading]]--
local data_config = {max_len = opt.maxLen, load_mode = opt.load_mode}
local ds = dp[opt.dataset](data_config)

--[[Model Construction]]--
if opt.xpPath ~= '' then
	assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
	if opt.cuda then
		require 'cunn'
		require 'optim'
		cutorch.setDevice(opt.useDevice)
	end
	xp = torch.load(opt.xpPath)
	agent = xp:model()
	local checksum = agent:parameters()[1]:sum()
	xp.opt.progress = opt.progress
	opt = xp.opt
else
	local nOutputFrame = (opt.maxLen - opt.convkw) / opt.convdw + 1
	local nOutputFrame = (nOutputFrame - opt.poolkw) / opt.pooldw + 1
	-- local nOutputFrame = (nOutputFrame - opt.convkw) / opt.convdw + 1
	-- local nOutputFrame = (nOutputFrame - opt.poolkw) / opt.pooldw + 1

	net = nn.Sequential()
	-- net:add(nn.WhiteNoise(0, 0.5))
	net:add(nn.TemporalConvolution(opt.inputC, opt.outputC, opt.convkw, opt.convdw))
	net:add(nn[opt.transfer]())
	net:add(nn.TemporalMaxPooling(opt.poolkw, opt.pooldw))
	-- net:add(nn.TemporalConvolution(opt.outputC, opt.outputC, opt.convkw, opt.convdw))
	-- net:add(nn[opt.transfer]())
	-- net:add(nn.TemporalMaxPooling(opt.poolkw, opt.pooldw))
	
	net:add(nn.SplitTable(1,2))
	net:add(nn.ReverseTable())

	-- recurrent layer
	local inputSize = opt.outputC
	for i, hiddenSize in ipairs(opt.rnnHiddenSize) do 

		-- if i ~= 1 and not opt.lstm then
		if not opt.lstm then
		  net:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
		end

		-- recurrent layer
		local rnn
		if opt.lstm then
			-- Long Short Term Memory
			rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
		else
			-- simple recurrent neural network
			rnn = nn.Recurrent(
				hiddenSize, -- first step will use nn.Add
				nn.Identity(), -- for efficiency (see above input layer) 
				nn.Linear(hiddenSize, hiddenSize), -- feedback layer (recurrence)
				nn[opt.transfer](), -- transfer function 
				opt.maxLen -- maximum number of time-steps per sequence
			)
			rnn = nn.Sequencer(rnn)
		end

		net:add(rnn)

		if opt.dropout then -- dropout it applied between recurrent layers
			net:add(nn.Sequencer(nn.Dropout(opt.dropout)))
		end
		
		inputSize = hiddenSize
	end
	local hiddenSize = opt.rnnHiddenSize[#opt.rnnHiddenSize]

	gater = nn.Sequential()
	gater:add(nn.Sequencer(nn.Linear(hiddenSize, 1))) 
	gater:add(nn.Sequencer(nn[opt.transfer]()))
	gater:add(nn.JoinTable(1, 1)) -- Join all activated hidden units
	gater:add(nn.SoftMax())

	attention = nn.ConcatTable()
	attention:add(gater)
	attention:add(nn.Sequencer(nn.Identity()))
	
	net:add(attention)
	net:add(nn.MixtureTable())
	-- net:add(nn.SelectTable(-1))
	net:add(nn.Linear(hiddenSize, 4))
	net:add(nn.LogSoftMax())

	if opt.uniform > 0 then
		for k,param in ipairs(net:parameters()) do
			param:uniform(-opt.uniform, opt.uniform)
		end
	end
end

--[[Propagators]]--
opt.decayFactor = (opt.minLR - opt.learningRate) / opt.saturateEpoch

train = dp.Optimizer{
	loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
	epoch_callback = function(model, report) -- called every epoch
		if report.epoch > 0 then
			opt.learningRate = opt.learningRate + opt.decayFactor
			opt.learningRate = math.max(opt.minLR, opt.learningRate)
			if not opt.silent then
				print("learningRate", opt.learningRate)
			end
		end
	end,
	callback = function(model, report)
		if opt.cutoffNorm > 0 then
			local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
			opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
			if opt.lastEpoch < report.epoch and not opt.silent then
				print("mean gradParam norm", opt.meanNorm)
			end
		end
		model:updateGradParameters(opt.momentum) -- affects gradParams
		model:updateParameters(opt.learningRate) -- affects params
		model:maxParamNorm(opt.maxOutNorm) -- affects params
		model:zeroGradParameters() -- affects gradParams
	end,
	feedback = dp.Confusion(),
	sampler = dp.ShuffleSampler{
		epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
	},
	progress = opt.progress
}

valid = dp.Evaluator{
	feedback = dp.Confusion(),
	sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
	progress = opt.progress
}

if not opt.noTest then
	tester = dp.Evaluator{
		feedback = dp.Confusion(),
		sampler = dp.Sampler{batch_size = opt.batchSize}
	}
end

xp = dp.Experiment{
	model = net,
	optimizer = train,
	validator = valid,
	tester = tester,
	-- random_seed = 10,
	random_seed = os.time(),
	observer = {
		dp.FileLogger(opt.logDir),
		dp.EarlyStopper{
			save_strategy = dp.SaveToFile{
				save_dir = opt.saveDir, 
				verbose = opt.silent
			},
			max_epochs = opt.maxTries,
			error_report={'validator','feedback','confusion','accuracy'},
			maximize = true
		}
	},
	max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
	print"Using CUDA"
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(opt.useDevice)
	xp:cuda()
else
	xp:float()
end

xp:verbose(not opt.silent)
if not opt.silent then
	print"TemporalCnn: "
	print(net)
end

xp.opt = opt

if checksum then
	assert(math.abs(xp:model():parameters()[1]:sum() - checksum) < 0.0001, "Loaded model parameters were changed???")
end

xp:run(ds)


