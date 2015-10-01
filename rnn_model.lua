require 'nngraph'
require 'util.OneHot'

--Just a single layer RNN for the time being
--TODO: Expand this to multi layer and add dropout
function create_rnn(input_vocab_size, rnn_hidden_size, output_vocab_size)
	inputs = {}
	outputs = {}

	local input = nn.Identity()() -- identity input node
	local prev_hidden = nn.Identity()() -- placeholder for previous hidden

	table.insert(inputs, input)
	table.insert(inputs, prev_hidden)

	local one_hot_input = OneHot(input_vocab_size)(input)
	local input_2_hidden = nn.Linear(input_vocab_size, rnn_hidden_size)(one_hot_input)
	local hidden_2_hidden = nn.Linear(rnn_hidden_size, rnn_hidden_size)(prev_hidden)
	local next_hidden = nn.Tanh()(nn.CAddTable(){input_2_hidden, hidden_2_hidden})
	
	--projection
	local projection = nn.Linear(rnn_hidden_size, output_vocab_size)(next_hidden)
	local logsoftmax = nn.LogSoftMax()(projection)

	table.insert(outputs, next_hidden)
	table.insert(outputs, logsoftmax)
	
	return nn.gModule(inputs, outputs)
end