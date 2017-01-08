require 'dp'
require 'loader'

local Tone, DataSource = torch.class("dp.Tone", "dp.DataSource")
Tone.isMnist = true

Tone._name = 'Tone'
Tone._text_axes = 'bwc'
Tone._classes = {1, 2, 3, 4}

function Tone:__init(config)
	config = config or {}
	assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
	local args, load_all, input_preprocess, target_preprocess

	args, self._valid_ratio, self._train_name, self._test_name, 
	self._valid_name, self._data_path, self._scale, self._binarize, 
	self._shuffle, load_all, input_preprocess, target_preprocess, self.max_len = xlua.unpack
	(
		{config},
		'Tone',
		'Tone classification problem.' ..
		'Note: Train and valid sets are already shuffled.',
		{	
			arg = 'valid_ratio',
			type = 'number',
			default = 1/6,
			help = 'proportion of training set to use for cross-validation.'
		},
		{ 
			arg = 'train_name',
			type = 'string',
			default = 'train',
			help = 'name of training set'
		},
		{
			arg = 'test_name',
			type = 'string',
			default = 'test_new',
			help = 'name of valid set'
		},
		{
			arg = 'valid_name',
			type = 'string',
			default = 'test',
			help = 'name of test set'
		},
		{
			arg = 'data_path',
			type = 'string',
			default = dp.DATA_DIR,
			help = 'path to data repository'
		},
		{
			arg = 'scale',
			type = 'table',
			help = 'bounds to scale the values between. [Default = {0,1}]'
			},
		{
			arg = 'binarize',
			type = 'boolean',
			help = 'binarize the inputs (0s and 1s)',
			default = false
		},
		{
			arg = 'shuffle',
			type = 'boolean',
			help = 'shuffle different sets',
			default = false
			},
		{
			arg = 'load_all',
			type = 'boolean',
			help = 'Load all datasets : train, valid, test.',
			default = true
		},

		{
			arg = 'input_preprocess',
			type = 'table | dp.Preprocess',
			help = 'to be performed on set inputs,measuring statistics '..
				'(fitting) on the train_set only, and reusing these to '..
				'preprocess the valid_set and test_set.'
		},
		{
			arg = 'target_preprocess',
			type = 'table | dp.Preprocess',
			help = 'to be performed on set targets, measuring statistics ' ..
				'(fitting) on the train_set only, and reusing these to ' ..
				'preprocess the valid_set and test_set.'
		},
		{
			arg = 'max_len',
			type = 'number',
			default = 128,
			help = 'max length of input data'
		}
	)
	self:loadTrain()
	self:loadValid()
	self:loadTest()
	DataSource.__init(self, {
		train_set = self:trainSet(), 
		valid_set = self:validSet(),
		test_set = self:testSet(), 
		input_preprocess = input_preprocess,
		target_preprocess = target_preprocess
	})
end

function Tone:loadTrain()
	local train_data = self:loadData(self._train_name)
	self:setTrainSet(
		self:createDataSet(train_data.data, train_data.label, 'train')
	)
	return self:trainSet()
end

function Tone:loadValid()
	local valid_data = self:loadData(self._valid_name)
	self:setValidSet(
		self:createDataSet(valid_data.data, valid_data.label, 'valid')
	)
	return self:validSet()
end

function Tone:loadTest()
	local test_data = self:loadData(self._test_name)
	self:setTestSet(
		self:createDataSet(test_data.data, test_data.label, 'test')
	)
	return self:testSet()
end

function Tone:createDataSet(inputs, targets, which_set)
	if self._shuffle then
	local indices = torch.randperm(inputs:size(1)):long()
		inputs = inputs:index(1, indices)
		targets = targets:index(1, indices)
	end
	
	if self._binarize then
		DataSource.binarize(inputs, 128)
	end

	-- construct inputs and targets dp.Views
	local input_v, target_v = dp.SequenceView(), dp.ClassView()
	input_v:forward(self._text_axes, inputs)
	target_v:forward('b', targets)
	target_v:setClasses(self._classes)
	-- construct dataset
	dataset = dp.DataSet{
		inputs = input_v, targets = target_v, which_set = which_set
	}
	return dataset
end

function Tone:loadData(name)
	local ldr = Loader:new(nil, self.max_len)
	local collection = {data = ldr.data[name]:float(), label = ldr.label[name]}
    function collection:size() 
        return self.data:size(1) 
    end
    for i = 1, 2 do
        -- we abandon engy...
        if i == 2 then
            collection.data[{{}, {}, i}]:fill(0)
            break
        end
        local std = collection.data[{{}, {}, i}]:std()
        collection.data[{{}, {}, i}]:div(std)
        -- for j = 1, collection:size() do
        --     local std = collection.data[{j, {}, i}]:std()
        --     collection.data[{j, {}, i}]:div(std)
        -- end
    end
    return collection
end
