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
	self._shuffle, load_all, input_preprocess, target_preprocess, self.max_len, self.load_mode = xlua.unpack
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
		},
		{
			arg = 'load_mode',
			type = 'string',
			default = 'linear',
			help = 'preprocessing data'
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
	local train_data = self:loadData(self._train_name, self.load_mode)
	self:setTrainSet(
		self:createDataSet(train_data.data, train_data.label, 'train')
	)
	return self:trainSet()
end

function Tone:loadValid()
	local valid_data = self:loadData(self._valid_name, self.load_mode)
	self:setValidSet(
		self:createDataSet(valid_data.data, valid_data.label, 'valid')
	)
	return self:validSet()
end

function Tone:loadTest()
	local test_data = self:loadData(self._test_name, self.load_mode)
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

function moving_avg(vec, window)
    local len = (#vec)[1]
    local new_vec = torch.Tensor(len):fill(0.0)
    for i = 1, (len - window) do
        local sum = vec[{{i, i + window - 1}}]:sum() / window
        new_vec[i] = sum
    end
    return new_vec
end

-- kick too large or too small values
function kick(vec, topsmall, toplarge)
    -- kick the smalls
    local new_vec = torch.Tensor()
    for i = 1, topsmall do
        vec:csub(vec:min())
        vec = vec[vec:gt(0.0)]:clone()
    end
    for i = 1, toplarge do
        _, inds = torch.max(vec, 1)
--         print(inds)
        for j = 1, (#inds)[1] do
            vec[inds[j]] = 0.0
        end
        vec = vec[vec:gt(0.0)]:clone()
    end
    return vec
end

-- fit a + b * x + c * x * x
local matrix = require "matrix"
local fit = {}
local function getresults( mtx )
    assert( #mtx+1 == #mtx[1], "Cannot calculate Results" )
    mtx:dogauss()
    -- tresults
    local cols = #mtx[1]
    local tres = {}
    for i = 1,#mtx do
        tres[i] = mtx[i][cols]
    end
    return unpack( tres )
end

function fit.parabola( x_values,y_values )
    -- x_values = { x1,x2,x3,...,xn }
    -- y_values = { y1,y2,y3,...,yn }

    -- values for A matrix
    local a_vals = {}
    -- values for Y vector
    local y_vals = {}

    for i,v in ipairs( x_values ) do
        a_vals[i] = { 1, v, v*v }
        y_vals[i] = { y_values[i] }
    end

    -- create both Matrixes
    local A = matrix:new( a_vals )
    local Y = matrix:new( y_vals )

    local ATA = matrix.mul( matrix.transpose(A), A )
    local ATY = matrix.mul( matrix.transpose(A), Y )

    local ATAATY = matrix.concath(ATA,ATY)

    return getresults( ATAATY )
end

function fit_quad(vec, targetLen)
    local len = (#vec)[1]
    local xs = {}
    local ys = {}
    local a, b, c
    for i = 1, len do
        table.insert(xs, i)
        table.insert(ys, vec[i])
    end
    a, b, c = fit.parabola(xs, ys)
    local res = torch.Tensor(targetLen)
    for i = 1, targetLen do
        local ii = (i - 1) / (targetLen - 1) * (len - 1) + 1
        res[i] = a + b * ii + c * ii * ii
    end
    return res
end

function make_linear_spine(vec, targetLen)
    local len = (#vec)[1]
    local ys = {}
    
    for i = 1, len do
        table.insert(ys, vec[i])
    end
    
    local res = torch.Tensor(targetLen)
    
    local i = 1
    for ind = 1, targetLen do
        local x = (ind - 1) * (len - 1) / (targetLen - 1) + 1
        if  x > i then
            i = i + 1
        end
        if x == i then 
            res[ind] = ys[i]
        else
            res[ind] = (ys[i] - ys[i - 1]) * (x - i + 1) + ys[i - 1]
        end
    end
    return res
end

function fit_tail(vec, targetLen, portion)
    local len = (#vec)[1]
    local start = math.modf(len * portion)
    local xs = {}
    local ys = {}
    local a, b, c
    for i = 1, len - start do
        table.insert(xs, i)
        table.insert(ys, vec[i + start])
    end
    a, b, c = fit.parabola(xs, ys)
    local res = torch.Tensor(targetLen)
    for i = 1, targetLen do
        local ii = (i - 1) / (targetLen - 1) * (len - start - 1) + 1
        res[i] = a + b * ii + c * ii * ii
    end
    return res
end

function Tone:loadData(name, mode)
	local ldr = Loader:new(nil, 256)
	local collection = {data = ldr.data[name]:float(), label = ldr.label[name]}
    function collection:size() 
        return self.data:size(1) 
    end
    
    -- Mel Frequency
    for j = 1, collection:size() do
        collection.data[{j, {}, 1}]:div(700):log1p():mul(5975.208)
    end

    -- local standardization
    for i = 1, 2 do
        for j = 1, collection:size() do
            local std = collection.data[{j, {}, i}]:std()
            collection.data[{j, {}, i}]:div(std)
        end
    end

    -- global standardization
    for i = 1, 2 do
        local std = collection.data[{{}, {}, i}]:std()
        collection.data[{{}, {}, i}]:div(std)
    end

    local resized_collection = {
    	data = torch.Tensor(collection:size(), self.max_len,2):float(),
    	label = collection.label
	}
    -- filter noice and shif to the beginning
    for i = 1, collection:size() do
        for j = 1, 256 do
            if (collection.data[{i, j, 2}] < 1.0) then
                collection.data[{i, j, 1}] = 0.0
                collection.data[{i, j, 2}] = 0.0
            end
        end
        local shifted_F0 = torch.Tensor(self.max_len):fill(0.0)
        local shifted_Engy = torch.Tensor(self.max_len):fill(0.0)
        local data = collection.data[{i, {}, {}}]
        local tmpF0 = data[{{}, 1}][data[{{}, 1}]:gt(0.0)]:clone()
        local tmpEngy = data[{{}, 2}][data[{{}, 2}]:gt(0.0)]:clone()
        
        tmpF0 = kick(tmpF0, 1, 0)
        tmpEngy = kick(tmpEngy, 1, 0)

        -- cut
        -- local cut_len = (#tmpF0)[1] * 0.4
        -- tmpF0 = tmpF0[{{cut_len, -1}}]

        -- smooth
        tmpF0 = moving_avg(tmpF0, 3)
        tmpF0 = tmpF0[tmpF0:gt(0.0)]:clone()
        tmpEngy = moving_avg(tmpEngy, 3)
        tmpEngy = tmpEngy[tmpEngy:gt(0.0)]:clone()
        
        if mode == 'shift' then
        	shifted_F0[{{2, (#tmpF0)[1] + 1}}] = tmpF0
        	shifted_Engy[{{2, (#tmpEngy)[1] + 1}}] = tmpEngy
        	resized_collection.data[{i, {}, 1}] = shifted_F0
        	resized_collection.data[{i, {}, 2}] = shifted_Engy
    	elseif mode == 'quad' then
        	resized_collection.data[{i, {}, 1}] = fit_quad(tmpF0, self.max_len)
        	resized_collection.data[{i, {}, 2}] = fit_quad(tmpEngy, self.max_len)
        elseif mode == 'linear' then
        	resized_collection.data[{i, {}, 1}] = make_linear_spine(tmpF0, self.max_len)
        	resized_collection.data[{i, {}, 2}] = make_linear_spine(tmpEngy, self.max_len)
        elseif mode == 'hybrid' then
            resized_collection.data[{i, {}, 1}] = fit_quad(tmpF0, self.max_len)
            resized_collection.data[{i, {}, 2}] = fit_tail(tmpF0, self.max_len, 0.5)
        else return nil end
    end
    -- restandardization on avg data
    for i = 1, 2 do
        for j = 1, collection:size() do
            local mean = resized_collection.data[{j, {}, i}]:mean()
            resized_collection.data[{j, {}, i}]:csub(mean)
        end
    end
    -- for i = 1, 2 do
    -- 	local mean = resized_collection.data[{{}, {}, i}]:mean()
    --     resized_collection.data[{{}, {}, i}]:csub(mean)
    --     -- local std = resized_collection.data[{{}, {}, i}]:std()
    --     -- resized_collection.data[{{}, {}, i}]:div(std)
    -- end
    -- we abandon engy...
    if mode ~= 'hybrid' then
        resized_collection.data[{{}, {}, 2}]:fill(0)
    end

    return resized_collection
end
