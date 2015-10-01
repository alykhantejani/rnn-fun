require 'nngraph'
local minibatch_loader = require 'util.CharSplitLMMinibatchLoader'
require 'util.OneHot'
require 'rnn_model'
require 'optim'


--we need to clone the rnn many times as
--each one has it's own output and gradInput
--variables (and are not overwritten by next time step)
function clone_many_times(model, times)
	local clones = {}
	
	local param, grad_param = model:getParameters()
	local p, grad_p = model:parameters()

	for t = 1, times do
		local clone = model:clone()
		local clone_p, clone_grad_p = clone:parameters()
		--set all clones to the same parameter view
		for i = 1, #p do
			clone_p[i]:set(p[i])
			clone_grad_p[i]:set(grad_p[i])
		end
		---------------------------------------------
		table.insert(clones, clone)
	end
	collectgarbage()
	return clones;
end



local seq_length = 10
local batch_size = 5
local loader = minibatch_loader.create('data/tinyshakespeare', batch_size, seq_length, {0.95, 0.05, 0})
local vocab_size = loader.vocab_size
local vocab = loader.vocab

print('vocab size: '..vocab_size)

--input_vocab_size, rnn_hidden_size, output_vocab_size
model = create_rnn(vocab_size, 128, vocab_size)
criterion = nn.ClassNLLCriterion()


cloned_models = clone_many_times(model, seq_length)
cloned_criteria = clone_many_times(criterion, seq_length)

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(cloned_models)

local h_init = torch.zeros(batch_size, 128)

function feval(x)
	if x ~= params then
		params:copy(x)
	end
	grad_params:zero()

	------------get minibatch----------------
	local input, target = loader:next_batch(1)

	local hidden_states = {}
	hidden_states[0] = torch.zeros(batch_size, 128)
	
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
	dhidden[seq_length] = torch.zeros(batch_size, 128) -- no error from the futuer hidden node

	for t = seq_length,1,-1 do
		local dcrit = cloned_criteria[t]:backward(predictions[t], target[{{}, t}])
		local dmodel = cloned_models[t]:backward({input[{{}, t}], hidden_states[t-1]},{dhidden[t], dcrit})
		dhidden[t-1] = dmodel[2] -- the error in the hidden state going into t-1
	end
	-------------------------------------------

	h_init = hidden_states[#hidden_states] -- update next batch initial state
	grad_params:div(seq_length)
	grad_params:clamp(-5, 5) -- clamp to avoid exploding gradients

	return loss, grad_params
end

config = {learningRate = 0.01, momentum = 0.9}
--This function updates the global parameters variable (which is a view on the models parameters)
optim.sgd(func_eval, parameters, config)