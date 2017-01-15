
require 'loader'
require 'nn'


print((require 'cjson'))

MAX_LEN = 256
max_len = 128
-- Loader:new(obj, max_len)
ldr = Loader:new(nil, MAX_LEN)
function moving_avg(vec, window)
    local len = (#vec)[1]
    local new_vec = torch.Tensor(len):fill(0.0)
    for i = 1, (len - window - 1) do
        local sum = vec[{{i, i + window - 1}}]:sum() / window
        new_vec[i+1] = sum
    end
    return new_vec
end
-- kick too large or too small values
function kick(vec, topsmall, toplarge)
    -- kick the smalls
    for i = 1, topsmall do
        vec:csub(vec:min())
        new_vec = vec[vec:gt(0.0)]:clone()
        vec = new_vec
    end
    for i = 1, toplarge do
        _, inds = torch.max(vec, 1)
--         print(inds)
--         for j = 1, (#inds)[1] do
--             vec[inds[j]] = 0.0
--         end
        vec[inds[1]] = 0.0
        vec[inds[-1]] = 0.0
        new_vec = vec[vec:gt(0.0)]:clone()
        vec = new_vec
    end
    return new_vec
end
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
    return res, {a, b, c}
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
    return res, {a, b, c}
end
function make_linear_spine(vec, targetLen)
    local len = (#vec)[1]
    local ys = {}
    
    for i = 1, len do
        table.insert(ys, vec[i])
    end
    
    local res
    local coef
    res, coef = torch.Tensor(targetLen)
    
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
    return res, coef
end
function make_dataset(name, mode)
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
        data = torch.Tensor(collection:size(), max_len,2):float(),
        label = collection.label
    }
    -- filter noice and shif to the beginning
    local coef = {}
    for i = 1, collection:size() do
        for j = 1, 256 do
            if (collection.data[{i, j, 2}] < 1.0) then
                collection.data[{i, j, 1}] = 0.0
                collection.data[{i, j, 2}] = 0.0
            end
        end
        local shifted_F0 = torch.Tensor(max_len):fill(0.0)
        local shifted_Engy = torch.Tensor(max_len):fill(0.0)
        local data = collection.data[{i, {}, {}}]
        local tmpF0 = data[{{}, 1}][data[{{}, 1}]:gt(0.0)]:clone()
        local tmpEngy = data[{{}, 2}][data[{{}, 2}]:gt(0.0)]:clone()
        
        peek = tmpF0
--         smooth (refer to a paper)
        
--         tmpF0 = tmpF0:index(1 ,torch.linspace((#tmpF0)[1],1,(#tmpF0)[1]):long())
        local tmpF0_prime = tmpF0:clone()
        local c_1 = 0.32
        local c_2 = 0.67
        -- half or double
        for j = 2, (#tmpF0)[1] do
            if math.abs(tmpF0[j] / 2 - tmpF0[j - 1]) < c_1 then 
                tmpF0[j] = tmpF0[j] / 2
            elseif math.abs(2 * tmpF0[j] - tmpF0[j - 1]) < c_1 then
                tmpF0[j] = tmpF0[j] * 2
            end
        end
        -- random error
        -- tmpF0[1] = (tmpF0[1] + tmpF0[2] + tmpF0[3]) /3
        -- tmpF0[2] = (tmpF0[1] + tmpF0[2] + tmpF0[3]) /3
        -- tmpF0[3] = (tmpF0[1] + tmpF0[2] + tmpF0[3]) /3
        
        for j = 3, (#tmpF0)[1] - 1 do
            if math.abs(tmpF0[j] - tmpF0_prime[j - 1]) > c_1 and math.abs(tmpF0[j + 1] - tmpF0_prime[j - 1]) > c_2 then
                tmpF0_prime[j] = 2 * tmpF0_prime[j - 1] - tmpF0_prime[j - 2]
            elseif math.abs(tmpF0[j] - tmpF0_prime[j - 1]) > c_1 and math.abs(tmpF0[j + 1] - tmpF0_prime[j - 1]) <= c_2 then
                tmpF0_prime[j] = 0.5 * (tmpF0_prime[j - 1] + tmpF0[j + 1])
            else 
                tmpF0_prime[j] = tmpF0[j] 
            end
        end
        -- backward check
        for j = (#tmpF0)[1] - 2, 2, -1 do
            if math.abs(tmpF0_prime[j] - tmpF0_prime[j + 1]) > c_1 and math.abs(tmpF0_prime[j - 1] - tmpF0_prime[j + 1]) > c_2 then
                tmpF0[j] = 2 * tmpF0_prime[j + 1] - tmpF0_prime[j + 2]
            elseif math.abs(tmpF0_prime[j] - tmpF0_prime[j - 1]) > c_1 and math.abs(tmpF0_prime[j + 1] - tmpF0_prime[j - 1]) <= c_2 then
                tmpF0[j] = 0.5 * (tmpF0_prime[j + 1] + tmpF0_prime[j - 1])
            else 
                tmpF0[j] = tmpF0_prime[j]
            end
            if math.abs(tmpF0_prime[j] - tmpF0[j]) < c_1 then
                for k = 1, j - 1 do
                    tmpF0[k] = tmpF0_prime[k]
                end
                break
            end
        end
--         tmpF0:index(1 ,torch.linspace((#tmpF0)[1],1,(#tmpF0)[1]):long())
        
--         tmpF0 = kick(tmpF0, 1, 0)
--         tmpEngy = kick(tmpEngy, 1, 0)
        
--         local cut_len = (#tmpF0)[1] * 0.25
--         tmpF0 = tmpF0[{{cut_len, -1}}]

        -- smooth
        tmpF0 = moving_avg(tmpF0, 5)
        tmpF0 = tmpF0[tmpF0:gt(0.0)]:clone()
        tmpEngy = moving_avg(tmpEngy, 5)
        tmpEngy = tmpEngy[tmpEngy:gt(0.0)]:clone()

        if mode == 'shift' then
            shifted_F0[{{2, (#tmpF0)[1] + 1}}] = tmpF0
            -- shifted_Engy[{{2, (#tmpEngy)[1] + 1}}] = tmpEngy
            resized_collection.data[{i, {}, 1}] = shifted_F0
            resized_collection.data[{i, {}, 2}] = shifted_Engy
        elseif mode == 'quad' then
            resized_collection.data[{i, {}, 1}], coef[i] = fit_quad(tmpF0, max_len)
            resized_collection.data[{i, {}, 2}] = fit_quad(tmpEngy, max_len)
        elseif mode == 'linear' then
            resized_collection.data[{i, {}, 1}] = make_linear_spine(tmpF0, max_len)
            resized_collection.data[{i, {}, 2}] = make_linear_spine(tmpEngy, max_len)
        end
    end
--     for i = 1, 2 do
--         for j = 1, dataset:size() do
--             local std = resized_collection.data[{j, {}, i}]:std()
--             resized_collection.data[{j, {}, i}]:div(std)
--         end
--     end
    for i = 1, 2 do
        for j = 1, collection:size() do
            local mean = resized_collection.data[{j, {}, i}]:mean()
            resized_collection.data[{j, {}, i}]:csub(mean)
--             local mean = resized_collection.data[{j, {}, i}][-1]
--             resized_collection.data[{j, {}, i}]:csub(mean)
--             local std = resized_collection.data[{{}, {}, i}]:std()
--             resized_collection.data[{{}, {}, i}]:div(std)
        end
    end
    for i = 1, 2 do
        local std = resized_collection.data[{{}, {}, i}]:std()
        resized_collection.data[{{}, {}, i}]:div(std)
    end
    
    resized_collection.data[{{}, {}, 2}]:fill(0.0)
    for i = 1, collection:size() do
        resized_collection[i] = {resized_collection.data[i], resized_collection.label[i]}
    end
    print("dataset "..name.." has been constructed!")
    return resized_collection, coef
end
trainset, tarincoef = make_dataset('train', 'linear')
testset, testcoef = make_dataset('test', 'linear')
newtestset, newtestcoef = make_dataset('test_new', 'linear')
function toTable(vec)
    local tmp = {}
    for i = 1, (#vec)[1] do
       tmp[i] = vec[i] 
    end
    return tmp
end

function makeJson(indata)
    local k = (#indata.data)[1]
    local aryDatas = {}
    for i = 1, k do
        aryDatas[i] = {}
        aryDatas[i]["label"] = indata.label[i]
        aryDatas[i]["input"] = toTable(indata.data[{i, {}, 1}])
    end
    return aryDatas
end
local jsonTable = {};
jsonTable['train'] = makeJson(trainset)
jsonTable['test'] = makeJson(testset)
jsonTable['test_new'] = makeJson(newtestset)
local jsonStr = (require 'cjson').encode(jsonTable)
print(jsonStr)