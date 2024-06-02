struct AttentionBlock: public torch::nn::Cloneable<AttentionBlock>
{
	AttentionBlock(int hidden_dim, int num_heads, int max_seq_len, float drop_out_rate)
	{
		lnorm1_ = register_module("attn_norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ hidden_dim }).elementwise_affine(true).eps(1e-06)));
		//mha_ = register_module("attn_mha", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(hidden_dim, num_heads).bias(true)));
		mha_wts_ = register_module("mha_wts", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, 3 * hidden_dim).bias(true)));
		mha_proj_ = register_module("mha_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim).bias(true)));


		lnorm2_ = register_module("attn_norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ hidden_dim }).elementwise_affine(true).eps(1e-06)));


		mlp_fc1_ = register_module("attn_mlp_fc1", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim).bias(true)));
		mlp_drop_out1_ = register_module("attn_mlp_drop_out1", torch::nn::Dropout(drop_out_rate));
		mlp_fc2_ = register_module("attn_mlp_fc2", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim).bias(true)));
		mlp_drop_out2_ = register_module("attn_mlp_drop_out2", torch::nn::Dropout(drop_out_rate));

		num_heads_ = num_heads;
		hidden_dim_ = hidden_dim;
	}

	void reset() {};

	torch::Tensor mlp(torch::Tensor x)
	{
		x = mlp_fc1_->forward(x);
		x = torch::gelu(x);
		x = mlp_drop_out1_->forward(x);
		x = mlp_fc2_->forward(x);
		x = mlp_drop_out2_->forward(x);
		return x;
	}

	torch::Tensor attention(torch::Tensor x, torch::Tensor mask)
	{
		torch::Tensor q;
		torch::Tensor k;
		torch::Tensor v;
		torch::Tensor y;
		torch::Tensor att;

		int B, T, C;

		B = x.sizes()[0];
		T = x.sizes()[1];
		C = x.sizes()[2];

		std::vector<torch::Tensor> splits = mha_wts_(x).split(hidden_dim_, 2);

		q = splits[0];
		k = splits[1];
		v = splits[2];

		q = q.view({ B, T, num_heads_, C / num_heads_ }).transpose(1, 2);
		k = k.view({ B, T, num_heads_, C / num_heads_ }).transpose(1, 2);
		v = v.view({ B, T, num_heads_, C / num_heads_ }).transpose(1, 2);

		att = q.matmul(k.transpose(-2, -1) * 1.0f / sqrt(k.size(-1)));


		att = att.masked_fill(mask, -INFINITY);

		att = torch::softmax(att, -1);

		y = att.matmul(v);

		y = y.transpose(1, 2).contiguous().view({ B, T, C });

		y = mha_proj_->forward(y);

		return y;



		return x;
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor mask)
	{
		torch::Tensor x_res;

		x_res = lnorm1_->forward(x);
		x = x + attention(x_res, mask);
		x = x + mlp(lnorm2_->forward(x));
		return x;
	}

	void to(const torch::Device device, bool non_blocking = false)
	{
		lnorm1_->to(device);
		//mha_->to(device);
		mha_wts_->to(device);
		mha_proj_->to(device);
		lnorm2_->to(device);
		mlp_fc1_->to(device);
		mlp_fc2_->to(device);
		mlp_drop_out1_->to(device);
		mlp_drop_out2_->to(device);
	}

	void train(bool on)
	{
		lnorm1_->train(on);
		//mha_->train(on);
		mha_wts_->train(on);
		mha_proj_->train(on);
		lnorm2_->train(on);
		mlp_fc1_->train(on);
		mlp_fc2_->train(on);
		mlp_drop_out1_->train(on);
		mlp_drop_out2_->train(on);
	}

private:
	torch::nn::LayerNorm lnorm1_{ nullptr };
	//torch::nn::MultiheadAttention mha_{ nullptr };
	torch::nn::Linear mha_wts_{ nullptr };
	torch::nn::Linear mha_proj_{ nullptr };

	torch::nn::LayerNorm lnorm2_{ nullptr };

	torch::nn::Linear mlp_fc1_{ nullptr };
	torch::nn::Dropout mlp_drop_out1_{ nullptr };
	torch::nn::Linear mlp_fc2_{ nullptr };
	torch::nn::Dropout mlp_drop_out2_{ nullptr };

	int num_heads_;
	int hidden_dim_;
};


class Net : public torch::nn::Cloneable<Net>
{
public:
	Net(uint32_t vocab_size, uint32_t max_seq_len, uint16_t num_heads, uint32_t hidden_dim, uint32_t num_layers, float drop_out_rate)
	{
		uint32_t i;
		char szName[100];

		tok_embeddings_ = register_module("tok_embeddings", torch::nn::Embedding(vocab_size, hidden_dim));
		pos_embeddings_ = register_module("pos_embeddings", torch::nn::Embedding(max_seq_len, hidden_dim));
		num_layers_ = num_layers;

		attn_block = new std::shared_ptr<AttentionBlock>[num_layers_];

		for (i = 0; i < num_layers_; i++)
		{
			sprintf_s(szName, sizeof(szName), "attn_%d", i);
			attn_block[i] = register_module<AttentionBlock>(szName, std::make_shared <AttentionBlock>(hidden_dim, num_heads, max_seq_len, drop_out_rate));
		}

		lnorm_ = register_module("lnorm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ hidden_dim }).elementwise_affine(true).eps(1e-06)));
		proj_ = register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, vocab_size).bias(true)));


		int64_t* temp = new int64_t[max_seq_len];

		for (i = 0; i < max_seq_len; i++)
		{
			temp[i] = i;
		}

		auto options_input = torch::TensorOptions().dtype(torch::kLong);
		pos_ = torch::from_blob(temp, { 1, max_seq_len }, options_input);
	}

	~Net() {}

	void reset() {};

	torch::Tensor forward(torch::Tensor x, torch::Tensor mask)
	{
		int i;
		torch::Tensor tok_emb;
		torch::Tensor pos_emb;

		tok_emb = tok_embeddings_->forward(x);
		pos_emb = pos_embeddings_->forward(pos_);

		x = tok_emb + pos_emb;

		for (i = 0; i < num_layers_; i++)
		{
			x = attn_block[i]->forward(x, mask);
		}

		x = lnorm_->forward(x);
		x = proj_->forward(x);

		return x;

	}

	void to(const torch::Device device, bool non_blocking = false)
	{
		int i;

		tok_embeddings_->to(device);
		pos_embeddings_->to(device);
		lnorm_->to(device);
		proj_->to(device);
		pos_ = pos_.to(device);

		for (i = 0; i < num_layers_; i++)
		{
			attn_block[i]->to(device);
		}

		pos_ = pos_.to(device);
	}

	void train(bool on)
	{
		int i;

		tok_embeddings_->train(on);
		pos_embeddings_->train(on);
		lnorm_->train(on);
		proj_->train(on);

		for (i = 0; i < num_layers_; i++)
		{
			attn_block[i]->train(on);
		}
	}

private:
	torch::nn::Embedding tok_embeddings_{ nullptr };
	torch::nn::Embedding pos_embeddings_{ nullptr };
	std::shared_ptr<AttentionBlock>* attn_block;
	torch::nn::LayerNorm lnorm_{ nullptr };
	torch::nn::Linear proj_{ nullptr };
	torch::Tensor pos_;
	uint32_t num_layers_;
};



// copied from light-tensor
class LearningRateSchedule
{
public:
	LearningRateSchedule() {}
	~LearningRateSchedule() {}

	virtual double GetLearningRate(uint64_t step, uint32_t epoch) = 0;

private:
	double base_lr_;
	double min_lr_;
	double max_lr_;
	uint64_t warmup_steps_;
	uint32_t warmup_epochs_;
	uint32_t total_epochs_;
	uint64_t steps_per_epoch_;
};

class LinearLearningRateSchedule : public LearningRateSchedule
{
public:
	LinearLearningRateSchedule(double min_lr, double max_lr, uint64_t steps_per_epoch, uint32_t total_epochs)
	{
		min_lr_ = min_lr;
		max_lr_ = max_lr;
		total_epochs_ = total_epochs;
		steps_per_epoch_ = steps_per_epoch;
	}

	double GetLearningRate(uint64_t step, uint32_t epoch = 0)
	{
		double scaler;
		double lr;

		scaler = step / (double)(steps_per_epoch_ * total_epochs_ - 1);

		lr = max_lr_ - (max_lr_ - min_lr_) * scaler;

		if (lr < min_lr_)
		{
			lr = min_lr_;
		}

		return lr;
	}

private:
	double min_lr_;
	double max_lr_;
	uint64_t steps_per_epoch_;
	uint32_t total_epochs_;

};


void PrintETA(double nseconds_latest_iteration, uint32_t remaining_iterations);
void Test(uint32_t max_seq_len, uint16_t num_heads, uint32_t hidden_dim, uint32_t num_layers);
