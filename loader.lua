require "torch"
local lfs = require "lfs"

-- Meta Class
Loader = {data = {}, width = 0, batch_size = 0, epoch_size = 0}

local function visit(path)
	local filelist = {}
	for file in lfs.dir(path) do
		if file ~= "." and file ~= ".." then
			local f = path..'/'..file
			local attr = lfs.attributes(f)
			assert(type(attr) == "table")
			if attr.mode == "file" then
				table.insert(filelist, f)
			end
		end
	end
	return filelist
end

local function prepare_index(path)
	local result = {}
	local pkgnamelist = {"train", "test", "test_new"}
	local label_word = {"one", "two", "three", "four"}
	for _, pkgname in ipairs(pkgnamelist) do
		local pkglist = {}
		for ind, labelname in ipairs(label_word) do
			filelist = visit(path..'/'..pkgname..'/'..labelname)
			for _, filename in ipairs(filelist) do
				if string.sub(filename, -3, -1) == ".f0" then
					table.insert(pkglist, {filename, string.gsub(filename, ".f0", ".engy"), ind})
				end
			end
		end
		result[pkgname] = pkglist
	end
	return result
end

local function readfile(file, size)
	result = torch.Tensor(size):zero()
	local i = 1
	for line in io.lines(file) do
		i = i + 1
		result[i] = line
	end
	return result
end

local function readentry(f0_engy_label, size)
	local X = torch.Tensor(size, 2)
	local f0 = f0_engy_label[1]
	local engy = f0_engy_label[2]
	local label = f0_engy_label[3]

	X[{{}, 1}] = readfile(f0, size)
	X[{{}, 2}] = readfile(engy, size)
	return X, label
end

local function readpkg(pkglist, size)
	local X = torch.Tensor(#pkglist, size, 2):fill(0.0)
	local y = torch.Tensor(#pkglist, 4):fill(0.0)
	for i, f0_engy_label in ipairs(pkglist) do
		local resX, y = readentry(f0_engy_label, size)
		X[i] = resX:clone()
	end
	return X, y
end

function Loader:new(o, width, batch_size, epoch_size)
	o = o or {}
	setmetatable(o, self)
	self.__index = self
	self.width = width or 0
	self.batch_size = batch_size or 0
	self.epoch_size = epoch_size or 0
	self.data = {}
	pkgres = prepare_index("data/toneclassifier")
	for name, pkglist in pairs(pkgres) do
		self.data[name] = readpkg(pkglist, width)
	end
end


Loader:new(nil, 256, 0, 0)
print(Loader.data['train']:size())
print(Loader.data['test']:size())
print(Loader.data['test_new']:size())







