#include <torch/torch.h>
#include "c10\cuda\CUDAFunctions.h"
#include <torch/nn/modules/conv.h>


#include "bootpin_tokenizer.h"

#include "tinystories_dataset_libtorch.h"
#include "libtorch_gpt.h"

torch::DeviceType device_type = torch::kCUDA;
//torch::DeviceType device_type = torch::kCPU;
torch::Device device(device_type, 0);

#include <windows.h>


int main()
{
	uint32_t batch_size = 12;
	uint32_t max_seq_len = 512;
	uint16_t num_heads = 8;
	uint32_t hidden_dim = 1024;
	uint32_t num_epochs = 5;
	uint32_t num_layers = 6;
	char filename[256];

	HMODULE hMod = LoadLibraryA("torch_cuda.dll");

	//Test(max_seq_len, num_heads, hidden_dim, num_layers); return 0;

	torch::Tensor x;
	torch::Tensor target;
	torch::Tensor self_attn_mask;
	torch::Tensor logits;
	torch::Tensor probs;
	torch::Tensor loss;
	torch::Tensor tokens_and_mask;
	float loss_val = 0;
	uint64_t total_iterations = 0;
	uint64_t avg_index = 0;
	uint32_t ignore_counts;
	int gradient_accumulation_steps;
	uint64_t num_examples_seen = 0;
	uint64_t num_training_examples;

	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;
	double nseconds_total = 0;
	double lr;


	uint32_t BOS;
	uint32_t EOS;
	uint32_t PAD;


	uint32_t vocabular_size;
	void* tokenizer;

	const char* file_name = "c:\\src\\bootpin_gpt\\data\\training\\TinyStories.csv";

	tokenizer = InitializeTokenizer("c:\\src\\bootpin_tokenizer\\data\\tokenizer.bin");

	vocabular_size = GetVocabularySize(tokenizer);

	BOS = vocabular_size;
	EOS = BOS + 1;
	PAD = EOS + 1;
	vocabular_size = PAD + 1;


	//auto custom_dataset = std::make_shared<CustomDataset>(file_name, max_seq_len, batch_size);
	//auto dataset = custom_dataset->map(torch::data::transforms::Stack<>());

	auto dataset = CustomDataset(file_name, max_seq_len, batch_size, tokenizer).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader( dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(batch_size));

	auto net = std::make_shared<Net>(vocabular_size, max_seq_len, num_heads, hidden_dim, num_layers, 0.0f);

	num_training_examples = dataset.size().value();

	torch::optim::AdamWOptions opt_adamw(3e-4);
	opt_adamw.weight_decay(5 * 1e-2);
	LinearLearningRateSchedule lrs(5e-5, 3e-4, num_training_examples, num_epochs);

	torch::optim::AdamW optimizer(net->parameters(), opt_adamw);

	net->to(device);

	net->train(true);

	int64_t param_count = 0;
	for (const auto& param : net->parameters())
	{
		param_count += param.numel();
	}
	printf("Number of parameters: %lld\n", param_count);

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(10258));


	gradient_accumulation_steps = 120 / batch_size;

	
	for (uint32_t epoch = 0; epoch < num_epochs; epoch++)
	{
		for (auto& batch : *data_loader)
		{			
			clock_begin = std::chrono::steady_clock::now();

			tokens_and_mask = batch.data;
			x = tokens_and_mask.index({ torch::indexing::Slice(), 0, torch::indexing::Slice() }).contiguous();
			self_attn_mask = tokens_and_mask.index({ torch::indexing::Slice(), torch::indexing::Slice(1, max_seq_len + 1), torch::indexing::Slice() }).contiguous();
			self_attn_mask = self_attn_mask.to(torch::kBool);

			x = x.to(device);
			self_attn_mask = self_attn_mask.to(device).unsqueeze(1);
			target = batch.target.to(device);

			logits = net->forward(x, self_attn_mask);

			probs = torch::nn::functional::log_softmax(logits, 2);

			loss = criterion(probs.view({ probs.sizes()[0] * probs.sizes()[1], probs.sizes()[2] }), target.view({ target.sizes()[0] * target.sizes()[1] }));
			loss = loss * (1.0f / gradient_accumulation_steps);
			loss.backward();

			num_examples_seen += batch_size;
			total_iterations++;

			if (!(total_iterations % gradient_accumulation_steps))
			{
				float floss = loss.item<float>() * gradient_accumulation_steps;
				loss_val += floss;
				lr = lrs.GetLearningRate(num_examples_seen - 1);
				printf("loss: %f avg loss: %f [epoch: %d iteration: %d / %d lr: %f]\n", floss, loss_val / (avg_index + 1), epoch, num_examples_seen, num_training_examples * num_epochs, lr);
				avg_index++;

				
				for (const auto& param_group : optimizer.param_groups())
				{
					((torch::optim::AdamWOptions&)param_group.options()).lr(lr);
				}
				
				/*
				for (auto& param_group : optimizer.param_groups()) 
				{
					param_group.options().set_lr(lr);
				}
				*/
				optimizer.step();
				optimizer.zero_grad();

				PrintETA(nseconds_total / num_examples_seen, num_training_examples * num_epochs - num_examples_seen);
				printf("\n");
			}

			if (!(total_iterations % 50000))
			{
				sprintf_s(filename, sizeof(filename), "c:\\src\\libtorch_gpt\\data\\tinystories_model_libtorch_shel_%d_%d_%d_%d_%lld.bin", max_seq_len, num_heads, hidden_dim, num_layers, total_iterations);
				torch::save(net, filename);
			}

			clock_end = std::chrono::steady_clock::now();
			time_span = clock_end - clock_begin;
			nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
			nseconds_total += nseconds;
		}
	}

	sprintf_s(filename, sizeof(filename), "c:\\src\\libtorch_gpt\\data\\tinystories_model_libtorch_shel_%d_%d_%d_%d_%lld.bin", max_seq_len, num_heads, hidden_dim, num_layers, total_iterations);
	torch::save(net, filename);
	return 0;
}


uint32_t tokens[1024];
void Test(uint32_t max_seq_len, uint16_t num_heads, uint32_t hidden_dim, uint32_t num_layers)
{
	int ret;
	int i;
	int j;
	int pad_len;
	uint32_t BOS;
	uint32_t EOS;
	uint32_t PAD;
	//uint32_t tokens[1024];
	float* self_attn_mask;

	torch::Tensor x;
	torch::Tensor mask;
	torch::Tensor x_nn;
	torch::Tensor mask_nn;
	torch::Tensor logits;
	torch::Tensor probs;
	uint32_t next_token;
	uint32_t total_tokens;
	char filename[256];
	uint32_t len;


	uint32_t vocabular_size;
	void* tokenizer;

	tokenizer = InitializeTokenizer("c:\\src\\bootpin_tokenizer\\data\\tokenizer.bin");

	vocabular_size = GetVocabularySize(tokenizer);

	BOS = vocabular_size;
	EOS = BOS + 1;
	PAD = EOS + 1;
	vocabular_size = PAD + 1;

	auto net = std::make_shared<Net>(vocabular_size, max_seq_len, num_heads, hidden_dim, num_layers, 0.0f);

	//sprintf_s(filename, sizeof(filename), "e:\\src\\bootpin_gpt\\data\\tinystories_model_s	hel_%d_%d_%d_%d.bin", max_seq_len, num_heads, hidden_dim, num_layers);
	sprintf_s(filename, sizeof(filename), "c:\\src\\libtorch_gpt\\data\\tinystories_model_libtorch_shel_512_8_1024_6_881920.bin", max_seq_len, num_heads, hidden_dim, num_layers);


	net->to(device);
	net->train(false);

	torch::load(net, filename);

	tokens[0] = BOS;
	len = sizeof(tokens) / sizeof(tokens[0]) - 1;
	ret = Encode(tokenizer, "", &tokens[1], &len);
	//ret = Encode(tokenizer, "Tom and Jane are friends. One day, Jane goes to Tom's house. Tom has a big pot of soup. He wants to share it with Jane. \"Jane, do you want some soup?\" Tom asks.", &tokens[1], &len);
	//ret = Encode(tokenizer, "Once upon a time, Tom and Jane went to the park.", &tokens[1], &len);
	//ret = Encode(tokenizer, "Alice and Jack walked up the street and met a girl in a red dress. The girl said to them, \"Hi, I'm Jane. What are your names?\"", &tokens[1], &len);
	//ret = Encode(tokenizer, "Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Anne's mom asked her, \"Anne, what is that you have in your left pocket?\"", &tokens[1], &len);
	//ret = Encode(tokenizer, "Lily wanted to get either a cat or a dog. Her mother didn't let her get a dog so instead she", &tokens[1], &len);
	//ret = Encode(tokenizer, "Lily wanted to get either a cat or a dog. Her mother didn't let her get a dog so instead she", &tokens[1], &len);
	//ret = Encode(tokenizer, "Words: call, leak, foolish\nFeatures : Dialogue\nSummary : Lily and Ben play with water in the garden, but don't notice the leak in the bucket. Mum calls them in to tell them they're wasting water and they learn to be more careful with it.\nStory :", &tokens[1], &len);
	//ret = Encode(tokenizer, "Words: point, stadium, fast\nSummary: Tom, Lily and their dad go to the stadium to watch cars race.They cheer for their favorite cars and have a fun time.\nFeatures : Dialogue\nStory:", &tokens[1], &len);


	len++; // accomodate BOS
	pad_len = max_seq_len - len;
	for (i = 0; i < pad_len; i++)
	{
		tokens[len + i] = PAD;
	}

	self_attn_mask = new float[max_seq_len * max_seq_len];


	x = torch::from_blob(tokens, { 1, max_seq_len }, torch::kInt);
	mask = torch::from_blob(self_attn_mask, { 1, max_seq_len, max_seq_len });

	total_tokens = len;
	while (true)
	{
		//
		// generate self attention mask
		//
		for (i = 0; i < max_seq_len; i++)
		{
			for (j = 0; j < max_seq_len; j++)
			{
				if (i < j)
				{
					self_attn_mask[i * max_seq_len + j] = 1; // look ahead masking
				}
				else
				{
					if (tokens[j] == PAD)
					{
						self_attn_mask[i * max_seq_len + j] = 1;
					}
					else
					{
						self_attn_mask[i * max_seq_len + j] = 0;
					}
				}
			}
		}

		//mask = mask.to(torch::kBool);
		x_nn = x.to(device);
		mask_nn = mask.to(device);
		mask_nn = mask_nn.to(torch::kBool);

		logits = net->forward(x_nn, mask_nn);
		probs = softmax(logits, -1);

		probs = probs.squeeze(0);
		probs = probs.to(torch::kCPU);


		//torch::Tensor val = torch::multinomial(probs, 1);
		//std::cout << val << "\n";


		//float* debug = (float*)probs.get_data_ptr();
		float* debug = (float*)(probs.contiguous().data_ptr());
		debug += (total_tokens - 1) * vocabular_size;
		float max_val = -1;
		int max_idx = 0;
		for (j = 0; j < vocabular_size; j++)
		{
			if (debug[j] > max_val)
			{
				max_val = debug[j];
				max_idx = j;
			}
		}


		//next_token = (uint32_t)((float*)val.get_data_ptr())[total_tokens-1];
		next_token = max_idx;

		tokens[total_tokens] = next_token;

		if (next_token == EOS)
		{
			break;
		}

		if (total_tokens >= max_seq_len)
		{
			break;
		}
		total_tokens++;

		unsigned int w_buffer_len = 5000;
		wchar_t w_buffer[5000];
		//Decode(tokenizer, &tokens[1], total_tokens - 1, w_buffer, &w_buffer_len);
		//std::wcout << w_buffer << std::endl;

	}

	unsigned int w_buffer_len = 10000;
	wchar_t* w_buffer = new wchar_t[w_buffer_len];

	Decode(tokenizer, &tokens[1], total_tokens - 1, w_buffer, &w_buffer_len);
	std::wcout << w_buffer << std::endl;

	printf("Len: %d\n", wcslen(w_buffer));
	printf("Tokens: %d\n", total_tokens);

	delete w_buffer;

	while (true)
	{

	}

}


void SpinForEver(const char* pszMessage)
{
	while (true)
	{
		printf("\r\n%s", pszMessage);
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}


void* BlockRealloc(void* current_block_ptr, uint64_t current_size, uint64_t new_size)
{
	unsigned char* reallocated_block_ptr;

	reallocated_block_ptr = new unsigned char[new_size];

	memcpy(reallocated_block_ptr, current_block_ptr, current_size);

	delete current_block_ptr;

	return reallocated_block_ptr;
}


