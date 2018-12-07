# %load bbb_nn.py
def construct_nn(ann_input, ann_output, n_input, n_output, n_train, n_hidden_1=64, n_hidden_2=32):
    # backwards compatibility
    return construct_nn_2lay(ann_input, ann_output, n_input, n_output, n_train, n_hidden_1, n_hidden_2)

def construct_nn_1lay(ann_input, ann_output, n_input, n_output, n_train, n_hidden_1=64):
    
    # Initialize random weights between each layer
    init_w_in_1 = np.random.randn(n_input, n_hidden_1).astype(theano.config.floatX)
    init_b_1 = np.random.randn(n_hidden_1).astype(theano.config.floatX)
    init_w_1_out = np.random.randn(n_hidden_1, n_output).astype(theano.config.floatX)
    
    with pm.Model() as neural_network:
        # Weights from input to first hidden layer
        w_in_1 = pm.Normal('w_in_1', mu=0, sd=0.1, shape=(n_input, n_hidden_1), testval=init_w_in_1)
        
        # Bias in first layer
        b_1 = pm.Normal('b_1',mu=0, sd=0.1, shape=(n_hidden_1), testval=init_b_1)
        
        # Weights from 1st hidden layer to output layer
        w_1_out = pm.Normal('w_1_out',mu=0, sd=0.1, shape=(n_hidden_1, n_output), testval=init_w_1_out)
        
        # Build neural-network using tanh activation function
        act_1 = pm.Deterministic('act_1',var=pm.math.tanh(pm.math.dot(ann_input,w_in_1)+b_1))        
        
        # Softmax is required at last layer
        act_out = pm.Deterministic('act_out',var=tt.nnet.softmax(pm.math.dot(act_1,w_1_out)))
        
        # Classification
        out = pm.Categorical('out',act_out,observed=ann_output,total_size=n_train)
        
    return neural_network

def construct_nn_2lay(ann_input, ann_output, n_input, n_output, n_train, n_hidden_1=64, n_hidden_2=32):
    
    # Initialize random weights between each layer
    init_w_1 = np.random.randn(n_input, n_hidden_1).astype(theano.config.floatX)
    init_w_2 = np.random.randn(n_hidden_1, n_hidden_2).astype(theano.config.floatX)
    init_out = np.random.randn(n_hidden_2, n_output).astype(theano.config.floatX)
    
    # Initialize bias for each layer  
    
    # did I ADd the bias wrong?
    init_b_1 = np.random.randn(n_hidden_1).astype(theano.config.floatX)
    init_b_2 = np.random.randn(n_hidden_2).astype(theano.config.floatX)
    
    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        w_0_1 = pm.Normal('w_0_1', mu=0, sd=0.15, shape=(n_input, n_hidden_1), testval=init_w_1)
        # Bias from in first layer
        b_1 = pm.Normal('b_1',mu=0, sd=0.15, shape=(n_hidden_1), testval=init_b_1)
        
        # Weights from 1st to 2nd layer
        w_1_2 = pm.Normal('w_1_2',mu=0, sd=0.15, shape=(n_hidden_1, n_hidden_2), testval=init_w_2)
        # Bias from in first layer
        b_2 = pm.Normal('b_2', mu=0, sd=0.15, shape=(n_hidden_2), testval=init_b_2)
        
        # Weights from hidden layer to output
        w_2_out = pm.Normal('w_2_out', mu=0, sd=0.15,shape=(n_hidden_2,n_output),testval=init_out)
        
        # Build neural-network using tanh activation function
        act_1 = pm.Deterministic('act_1',var=pm.math.tanh(pm.math.dot(ann_input,w_0_1)+b_1))
        act_2 = pm.Deterministic('act_2',var=pm.math.tanh(pm.math.dot(act_1,w_1_2)+b_2))
        
        # Softmax is required at last layer
        act_out = pm.Deterministic('act_out',var=tt.nnet.softmax(pm.math.dot(act_2,w_2_out)))
        
        # Classification
        out = pm.Categorical('out',act_out,observed=ann_output,total_size=n_train)
        
    return neural_network