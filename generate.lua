require 'torch'
require 'rnn_model'

---------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate Text from a snapshot of a model')
cmd:text()
-------------------------required------------------------------
cmd:argument('-snapshot', 'snapshot to use to sample from')
-------------------------optional------------------------------
cmd:option('-seed_text', '', 'text to seed from')
cmd:option('-seed', 123, 'random seed')
cmd:option('-temperature', 1, 'If set will use sampling with set temperature instead of choosing max prediction')
cmd:option('-max_length', 140, 'max length of generated text')
---------------------------------------------------------------

params = cmd:parse(arg)

print(string.format('setting random seed to %d', params.seed))
torch.manualSeed(params.seed)

local seed_text = params.seed_text

print('loading snapshot...')
local snapshot = torch.load(params.snapshot)

local temperature = params.temperature
local model = snapshot.model
local char_to_idx = snapshot.vocab_mapping
local idx_to_char = {}

for char, idx in pairs(char_to_idx) do
	idx_to_char[idx] = char
end

model:evaluate() -- for dropout if used
local hidden_state = torch.zeros(1, snapshot.input_hidden_size):double()
local predictions
local generated_text = ""
local text_length = 1
local max_text_length = params.max_length

if #seed_text > 0 then
	print(string.format('using seed text %s', seed_text))
	for char in seed_text:gmatch('.') do 
		local input = torch.Tensor(1):fill(char_to_idx[char])
		generated_text = generated_text..char
		local output = model:forward{input, hidden_state}
		hidden_state = output[1]
		predictions = output[2]
		text_length = text_length + 1
	end
else
	predictions = torch.Tensor(1, #idx_to_char):fill(1.0/#idx_to_char) --uniform probability
end

local val, char_idx

for i = text_length, max_text_length do 

	if temperature == 0 then
		--usr argmax
		al, char_idx = predictions:max(2)
		char_idx = char_idx:resize(1)
	end
	local probs = torch.exp(predictions):squeeze()
	probs:div(torch.sum(probs)) -- renormalize so probs sum to one
	char_idx = torch.multinomial(probs:float(), 1):resize(1):float()

	generated_text = generated_text..idx_to_char[char_idx[1]]
	local output = model:forward{char_idx, hidden_state}
	hidden_state = output[1]
	predictions = output[2]
end

print('generated output is:')
print('')
print(generated_text)