require 'torch'
require 'nngraph'
require 'utils.OneHot'
require 'rnn_model'
require 'optim'
local minibatch_loader = require 'utils.CharSplitLMMinibatchLoader'
local model_utils = require 'utils.model_utils'
local path = require 'pl.path'

-------------------------------------------------------------------
----------------------Command Line Params--------------------------
-------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Arguments')
cmd:argument('-data_dir','txt file with data')
cmd:text('Options')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
cmd:option('-batch_size', 64, 'batch size')
cmd:option('-max_epoch', 10, 'max epochs to train for')
cmd:option('-sequence_length', 10, 'sequence length to train on')
cmd:option('-num_hidden_units', 128, 'number of hidden units')
cmd:option('-display_iteration', 20, 'when to display training accuracy (1 iteration = 1 batch)')
cmd:option('-validation_iteration', 20, 'when to display validation accuracy (1 iteration = 1 batch')
-------------------Optimization----------------------------------
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
-------------------------Snapshotting--------------------------------
cmd:option('-snapshot', '', 'snapshot to begin training from')
cmd:option('-snapshot_dir', './snapshots/', 'snapshot directory')
cmd:option('-snapshot_epoch', 1, 'snapshot after how many epochs?')
----------------------------Misc------------------------------------
cmd:option('-gpu_id', -1, 'gpu ID to train on')
cmd:option('-log', '', 'output log file')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-seed',123,'torch manual random number generator seed')


local params = cmd:parse(arg)

if params.log ~= '' then
	cmd:log(params.log, params)
end

torch.manualSeed(params.seed)

-------------------------------------------------------------------
----------------------Initialize Variables-------------------------
-------------------------------------------------------------------
local max_epochs = params.max_epoch
local clip = params.grad_clip
local num_hidden_units = params.num_hidden_units
local display_iteration = params.display_iteration
local validation_iteration = params.validation_iteration
local resume_from_snapshot = (params.snapshot ~= '')
----------Optim--------------------------------------------------
local learning_rate = params.learning_rate
local decay_rate = params.decay_rate
local learning_rate_decay = params.learning_rate_decay
local learning_rate_decay_after = params.learning_rate_decay_after
------------Snapshots-------------------------------------------
local snapshot_file = params.snapshot
local snapshot_dir = params.snapshot_dir
local snapshot_epoch = params.snapshot_epoch
if not path.exists(params.snapshot_dir) then
	print('creating snapshot directory ' .. params.snapshot_dir)
	path.mkdir(params.snapshot_dir)
end
------------Loader---------------------------------------------
local seq_length = params.sequence_length
local batch_size = params.batch_size
local test_frac = math.max(0, 1 - (params.train_frac + params.val_frac))
local split_sizes = {params.train_frac, params.val_frac, test_frac} 
local loader = minibatch_loader.create(params.data_dir, batch_size, seq_length, split_sizes)
local vocab_size = loader.vocab_size
local vocab = loader.vocab
print('loaded dataset with ' .. vocab_size .. ' unique characters')
------------GPU---------------------------------------------
local gpu_id = params.gpu_id
local gpu = gpu_id ~= -1

if gpu then
	require 'cutorch'
	require 'cunn'
	cutorch.manualSeed(params.seed)
end

local model = create_rnn(vocab_size, num_hidden_units, vocab_size)
local criterion = nn.ClassNLLCriterion()

if resume_from_snapshot then
	print(string.format('loading snapshot from file', snapshot_file))
	local snapshot = torch.load(snapshot_file)
	model = snapshot.model
	for char, idx in pairs(snapshot.vocab_mapping) do
		if loader.vocab_mapping[char] ~= idx then
			print(string.format('ERROR: and char %s does not have the same index in the loader (%d) and snapshot (%d)', char, loader.vocab_mapping[char], idx))
			exit(1)
		end
	end
end

if gpu then
	model = model:cuda()
	criterion = criterion:cuda()
end

local cloned_models = model_utils.clone_many_times(model, seq_length)
local cloned_criteria = model_utils.clone_many_times(criterion, seq_length)

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(unpack(cloned_models))
params:uniform(-0.08, 0.08) --random initialization

local h_init = torch.zeros(batch_size, num_hidden_units)
if gpu then
	h_init = h_init:cuda()
end

function evaluate_validation_set(max_batches)
	local num_batches = loader.split_sizes[2] -- validation set
	if max_batches then 
		max_batches = math.min(max_batches,  num_batches)
	else
		max_batches = num_batches
	end

	loader:reset_batch_pointer(2) -- move batch iteration pointer for this split to front	

	local hidden_states = {}
	hidden_states[0] = torch.zeros(batch_size, num_hidden_units)

	if gpu then 
		hidden_states[0] = hidden_states[0]:cuda()
	end

	local predictions = {}
	local loss = 0

	for i = 1, max_batches do
		local input, target = loader:next_batch(2)
		if gpu then 
			input = input:float():cuda()
			target = target:float():cuda()
		end
		for t = 1, seq_length do
			cloned_models[t]:evaluate()
			local result = cloned_models[t]:forward({input[{{}, t}], hidden_states[t-1]})
			predictions[t] = result[2] -- log softmax output
			local next_hidden = result[1] --hidden output
			hidden_states[t] = next_hidden
			loss = loss + cloned_criteria[t]:forward(predictions[t], target[{{}, t}])
		end
		loss = loss / seq_length
		hidden_states[0] = hidden_states[#hidden_states] -- carry over hidden state
	end
	loss = loss / max_batches

	return loss
end


-------------------------------------------------------------------------
---------------------------Main loop-------------------------------------
-------------------------------------------------------------------------
function feval(x)
	if x ~= params then
		params:copy(x)
	end
	grad_params:zero()

	------------get minibatch----------------
	local input, target = loader:next_batch(1) -- training_batch
	
	if gpu then
		input = input:cuda()
		target = target:cuda()
	end

	local hidden_states = {}
	hidden_states[0] = h_init
	
	local predictions = {}
	local loss = 0
	
	-----------Forward Pass-------------------
	for t=1,seq_length do
		local result = cloned_models[t]:forward{input[{{},t}], hidden_states[t-1]} -- feed forward the tth character for a whole batch
		predictions[t] = result[2] -- log softmax output
		local next_hidden = result[1] --hidden output
		hidden_states[t] = next_hidden
		err = cloned_criteria[t]:forward(predictions[t], target[{{}, t}])
		loss = loss + err
	end
	loss = loss / seq_length

	-------------Backward Pass-----------------
	--make error at time t 0 as there is no influence from the future
	local dhidden = {}
	dhidden[seq_length] = torch.zeros(batch_size, num_hidden_units) -- no error from the futuer hidden node

	for t = seq_length,1,-1 do
		local dcrit = cloned_criteria[t]:backward(predictions[t], target[{{}, t}])
		local dmodel = cloned_models[t]:backward({input[{{}, t}], hidden_states[t-1]},{dhidden[t], dcrit})
		dhidden[t-1] = dmodel[2] -- the error in the hidden state going into t-1
	end
	-------------------------------------------

	h_init = hidden_states[#hidden_states] -- update next batch initial state
	grad_params:div(seq_length)
	grad_params:clamp(-clip, clip) -- clamp to avoid exploding gradients

	return loss, grad_params
end

function train()
	local optim_state = {learningRate = learning_rate, alpha = decay_rate}
	local timer = torch.Timer()

	local iterations_per_epoch = loader.ntrain

	local train_losses = {}
	local val_losses = {}

	local max_iterations = max_epochs * loader.ntrain
	local start_iteration = 1

	local start_epoch = 1

	if resume_from_snapshot then
		print('resuming training from snapshot ' .. snapshot)
		local snapshot = torch.load(snapshot)
		train_losses = snapshot.train_losses
		val_losses = snapshot.val_losses
		start_epoch = snapshot.epoch + 1
		optim_state = snapshot.optim_state
	end

	for epoch = start_epoch, max_epochs do
		print(string.format('Starting Epoch [%d]', epoch))

		for iter = 1, iterations_per_epoch do
			for t = 1, #cloned_models do
				cloned_models[t]:training()
			end

			timer:reset()
			p, cost = optim.rmsprop(feval, params, optim_state)
			loss = cost[1]

			train_losses[(iterations_per_epoch * (epoch - 1)) + iter] = loss

			if (iter % display_iteration) == 0 then
				print(string.format('Epoch [%d] Iteration [%d/%d]: Loss = %6.8f (this batch took %.4f ms)', epoch, iter, iterations_per_epoch, loss, (timer:time().real * 1000.0)))
			end
			if (iter % validation_iteration) == 0 then
				local val_loss = evaluate_validation_set()

				val_losses[(iterations_per_epoch * (epoch - 1)) + iter] = val_loss
				print(string.format('[Validation Summary] Iteration [%d]: %.4f', iter, val_loss))
			end
		end
		
		-- exponential learning rate decay
	    if learning_rate_decay < 1 then
	        if epoch >= learning_rate_decay_after then
	            local decay_factor = learning_rate_decay
	            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
	            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
	        end
	    end
	    ---------------------------------------------

	    --Snapshot-----------------------------------
	    if epoch % snapshot_epoch == 0 then
			local out_path = path.join(snapshot_dir, 'snapshot_epoch' .. epoch .. '.t7')
			local snapshot = {}
			snapshot.model = cloned_models[1]
			snapshot.input_hidden_size = num_hidden_units
			snapshot.optim_state = optim_state
			snapshot.train_losses = train_losses
			snapshot.val_losses = val_losses
			snapshot.vocab_mapping = loader.vocab_mapping
			snapshot.epoch = epoch

			print(string.format('saving snapshot to location %s', out_path))
			torch.save(out_path, snapshot)
	    end
	    ------------------------------------------------
	end
end

train()