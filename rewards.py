import tensorflow as tf

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
	"""
	Compute diversity reward and representativeness reward

	Args:
		seq: sequence of features, shape (1, seq_len, dim)
		actions: binary action sequence, shape (1, seq_len, 1)
		ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
		temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
		use_gpu (bool): whether to use GPU
	"""
	# _seq = tf.stop_gradient(seq)
	# _actions = tf.stop_gradient(actions)
	_seq = seq
	_actions = actions
	pick_idxs = tf.squeeze(tf.where(tf.math.not_equal(tf.squeeze(_actions), tf.constant(0, dtype=tf.int32))))
	num_picks = len(pick_idxs) if len(pick_idxs.shape) > 0 else 1
	
	if num_picks == 0:
		# give zero reward is no frames are selected
		reward = tf.constant(0.0)
		# if use_gpu: reward = reward.cuda()
		return reward

	_seq = tf.squeeze(_seq)
	n = _seq.shape[0]

	# compute diversity reward
	pick_idxs = pick_idxs.numpy()
	if num_picks == 1:
		reward_div = tf.constant(0.0)
		# if use_gpu: reward_div = reward_div.cuda()
	else:
		normed_seq = _seq / tf.linalg.norm(_seq, axis=1, keepdims=True)
		dissim_mat = 1.0 - tf.linalg.matmul(normed_seq, tf.transpose(normed_seq)) # dissimilarity matrix [Eq.4]

		dissim_submat = dissim_mat.numpy()[pick_idxs,:][:,pick_idxs]
		if ignore_far_sim:
			# ignore temporally distant similarity
			pick_idxs = tf.convert_to_tensor(pick_idxs)
			pick_mat = tf.broadcast_to(pick_idxs, (num_picks, num_picks))
			temp_dist_mat = tf.math.abs(pick_mat - tf.transpose(pick_mat))
			temp_dist_mat = temp_dist_mat.numpy()
			dissim_submat[temp_dist_mat > temp_dist_thre] = 1.0
		reward_div = tf.math.reduce_sum(dissim_submat) / (num_picks * (num_picks - 1.0)) # diversity reward [Eq.3]

	# compute representativeness reward
	dist_mat = tf.broadcast_to(tf.math.reduce_sum(tf.math.pow(_seq, 2), axis=1, keepdims=True), (n, n))
	dist_mat = dist_mat + tf.transpose(dist_mat)
	# dist_mat.addmm_(1, -2, _seq, _seq.t())
	dist_mat = dist_mat - 2*tf.linalg.matmul(_seq, _seq, False, True)
	pick_idxs = pick_idxs.numpy()
	dist_mat = tf.convert_to_tensor(dist_mat.numpy()[:,pick_idxs])
	dist_mat = tf.math.reduce_min(dist_mat, axis=1, keepdims=True)[0]
	#reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
	reward_rep = tf.math.exp(-tf.math.reduce_mean(dist_mat))

	# combine the two rewards
	reward = (reward_div + reward_rep) * 0.5

	return reward
