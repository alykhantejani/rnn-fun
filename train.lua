require 'torch'
require 'nngraph'
require 'utils.OneHot'
require 'rnn_model'
require 'optim'
local minibatch_loader = require 'utils.CharSplitLMMinibatchLoader'
local model_utils = require 'utils.model_utils'

-------------------------------------------------------------------
----------------------Command Line Params--------------------------
-------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Arguments')
cmd:argument('-data_dir','txt file with data')
cmd:text('Options')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
cmd:option('-batch_size', 64, 'batch size')
cmd:option('-max_epoch', 10, 'max epochs to train for')
cmd:option('-sequence_length', 10, 'sequence length to train on')
cmd:option('-num_hidden_units', 128, 'number of hidden units')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-weights', '', 'pretrained model to begin training from')
cmd:option('-display_iteration', 20, 'when to display training accuracy (1 iteration = 1 batch)')
cmd:option('-test_iteration', 20, 'when to display validation accuracy (1 iteration = 1 batch')
cmd:option('-optim_state','','path to the optimiser state file')
cmd:option('-snapshot_dir', './snapshots/', 'snapshot directory')
cmd:option('-snapshot_iteration', 1, 'snapshot after how many epochs?')
cmd:option('-gpu_id', -1, 'gpu ID to train on')
cmd:option('-log', '', 'output log file')

local params = cmd:parse(arg)

if params.log ~= '' then
	cmd:log(params.log, params)
end

-------------------------------------------------------------------
----------------------Initialize Variables-------------------------
-------------------------------------------------------------------

local seq_length = params.sequence_length
local batch_size = params.batch_size
local max_epochs = params.max_epoch
local clip = params.grad_clip
local num_hidden_units = params.num_hidden_units
local display_iteration = params.display_iteration
local test_frac = math.max(0, 1 - (params.train_frac + params.val_frac))
local split_sizes = {params.train_frac, params.val_frac, test_frac} 

local loader = minibatch_loader.create(params.data_dir, batch_size, seq_length, split_sizes)
local vocab_size = loader.vocab_size
local vocab = loader.vocab

print('loaded dataset with ' .. vocab_size .. ' unique characters')

--input_vocab_size, rnn_hidden_size, output_vocab_size
local model = create_rnn(vocab_size, num_hidden_units, vocab_size)
local criterion = nn.ClassNLLCriterion()

local cloned_models = model_utils.clone_many_times(model, seq_length)
local cloned_criteria = model_utils.clone_many_times(criterion, seq_length)

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(unpack(cloned_models))

local h_init = torch.zeros(batch_size, num_hidden_units)

function feval(x)
	if x ~= params then
		params:copy(x)
	end
	grad_params:zero()

	------------get minibatch----------------
	local input, target = loader:next_batch(1)

	local hidden_states = {}
	hidden_states[0] = torch.zeros(batch_size, num_hidden_units)
	
	local predictions = {}
	local loss = 0
	
	-----------Forward Pass-------------------
	for t=1,seq_length do
		result = cloned_models[t]:forward{input[{{},t}], hidden_states[t-1]} -- feed forward the tth character for a whole batch
		predictions[t] = result[2] -- log softmax output
		next_hidden = result[1] --hidden output
		hidden_states[t] = next_hidden
		loss = loss + cloned_criteria[t]:forward(predictions[t], target[{{}, t}])
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

-------------------------------------------------------------------------
---------------------------Main loop-------------------------------------
-------------------------------------------------------------------------
local optim_state = {learningRate = params.learning_rate, alpha = params.decay_rate}
local timer = torch.Timer()

for epoch = 1, max_epochs do
	print(string.format('Starting Epoch [%d]', epoch))
	for iter = 1, loader.ntrain do
		timer:reset()
		p, cost = optim.rmsprop(feval, params, optim_state)
		loss = cost[1]
		if (iter % display_iteration) == 0 then
			print(string.format('Epoch [%d] Iteration [%d]: Loss = %.4f (this batch took %.4f ms)', epoch, iter, loss, (timer:time().real * 1000.0)))
		end
	end
end
