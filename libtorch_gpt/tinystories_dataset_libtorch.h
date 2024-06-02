void SpinForEver(const char* pszMessage);
void* BlockRealloc(void* current_block_ptr, uint64_t current_size, uint64_t new_size);


struct CustomDataset : torch::data::datasets::Dataset<CustomDataset>
{
	CustomDataset(const char* file_name, int max_context_len, int batch_size, void* tokenizer)
	{
		bool ret;
		FILE* stream;
		int byte;
		uint64_t count;
		char temp[temp_buffer_size];
		size_t len;
		size_t max_len;
		bool copy;
		uint32_t block_allocate_stride;


		tokenizer_ = tokenizer;

		BOS_ = ::GetVocabularySize(tokenizer_);
		EOS_ = BOS_ + 1;
		PAD_ = EOS_ + 1;
		vocabular_size_ = PAD_ + 1;


		max_context_len_ = max_context_len;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		errno_t err;
		err = fopen_s(&stream, file_name, "rb");
		if (err)
		{
			ret = false;
			goto Exit;
		}
#else
		stream = fopen(file_name, "rb");
		if (!stream)
		{
			ret = -1;
			goto Exit;
		}
#endif


		text_examples_ = nullptr;
		num_training_examples_ = 0;
		count = 0;
		copy = false;
		max_len = 0;
		memset(temp, 0, sizeof(temp));
		block_allocate_stride = 1000000;


		while ((byte = fgetc(stream)) != EOF)
		{
			if (copy)
			{
				temp[len++] = (char)byte;
				if (len >= sizeof(temp))
				{
					SpinForEver("Buffer too small error loading dataset!\n");
				}
			}

			if ((char)byte == '"')
			{
				if (copy)
				{
					byte = fgetc(stream); // see if the next character is another '"', if it is, then this is not a delimiter but quotes in the text
					if (byte == EOF)
					{
						ret = false;
						goto Exit;
					}
					count++;
					if ((char)byte != '"')
					{
						len--; // get rid of last " which is just the record delimiter

						if (len > max_len)
						{
							max_len = len;
						}

						copy = false;
						temp[len] = '\0';


						if (!(num_training_examples_ % block_allocate_stride))
						{
							text_examples_ = (char**)BlockRealloc(text_examples_, sizeof(char*) * (num_training_examples_), sizeof(char*) * (num_training_examples_ + block_allocate_stride));
						}

						if (len < 70)
						{
							continue;
						}

						text_examples_[num_training_examples_] = new char[len + 1];
						strcpy_s(text_examples_[num_training_examples_], len + 1, temp);

						len = strlen(text_examples_[num_training_examples_]);
						for (int k = 0; k < len; k++)
						{
							if (text_examples_[num_training_examples_][k] == '’')
							{
								text_examples_[num_training_examples_][k] = '\'';
							}
						}

						num_training_examples_++;
					}
				}
				else
				{
					copy = true;
					len = 0;
				}
			}

			count++;
		}

		get_item_retries_ = 0;

	Exit:

		printf("PAD: %d\n", PAD_);
		return;
	}

	torch::data::Example<> get(size_t index) override
	{
		uint32_t* tokens_ptr;
		uint32_t* attn_mask_ptr;
		int64_t* target_ptr;
		
		uint32_t len;
		uint32_t char_len;
		uint32_t pad_len;
		uint32_t i;
		uint32_t j;
		int ret;
		torch::Tensor contig_tokens;
		torch::Tensor contig_target;

		torch::Tensor tokens;
		torch::Tensor target;


		tokens = torch::empty({ max_context_len_ + 1,  max_context_len_ }, torch::kInt); // don't know how to return 3 tensors, so returning tokens and mask together
		target = torch::empty({ max_context_len_ }, torch::kLong);


		contig_tokens = tokens.contiguous();
		contig_target = target.contiguous();

		tokens_ptr = (uint32_t*)contig_tokens.data_ptr();
		attn_mask_ptr = tokens_ptr + max_context_len_;
		target_ptr = (int64_t*)contig_target.data_ptr();

		while (true)
		{
			char_len = strlen(text_examples_[index]);
			if (char_len < 10)
			{
				index = rand() % num_training_examples_;
				continue;
			}

			tokens_ptr[0] = BOS_;
			len = max_context_len_ - 1;
			ret = Encode(tokenizer_, text_examples_[index], &tokens_ptr[1], &len);
			if (ret || (len > char_len))
			{
				index = rand() % num_training_examples_;
			}
			else
			{
				break;
			}
		}

		for (i = 0; i < len; i++)
		{
			target_ptr[i] = tokens_ptr[i + 1];
		}
		target_ptr[len] = EOS_;


		len++; // accomodate BOS_/EOS_
		pad_len = max_context_len_ - len;
		for (i = 0; i < pad_len; i++)
		{
			tokens_ptr[len + i] = PAD_;
			target_ptr[len + i] = PAD_;
		}


		//
		// generate self attention mask
		//
		for (i = 0; i < max_context_len_; i++)
		{
			for (j = 0; j < max_context_len_; j++)
			{
				if (i < j)
				{
					attn_mask_ptr[i * max_context_len_ + j] = 1; // look ahead masking
				}
				else
				{
					if (tokens_ptr[j] == PAD_)
					{
						attn_mask_ptr[i * max_context_len_ + j] = 1;
					}
					else
					{
						attn_mask_ptr[i * max_context_len_ + j] = 0;
					}
				}
			}
		}

		return { tokens, target };
	}
	/*
	torch::data::Example<> get(size_t index) override
	{
		uint32_t* tokens_ptr;
		int64_t* target_ptr;
		float* self_attn_mask;
		uint32_t len;
		uint32_t char_len;
		uint32_t pad_len;
		uint32_t i;
		uint32_t j;
		int ret;
		torch::Tensor contig_tokens;
		torch::Tensor contig_target;

		torch::Tensor tokens;
		torch::Tensor target;


		tokens = torch::empty({ max_context_len_ }, torch::kInt);
		target = torch::empty({ max_context_len_, 1 }, torch::kLong);


		contig_tokens = tokens.contiguous();
		contig_target = target.contiguous();

		tokens_ptr = (uint32_t*)contig_tokens.data_ptr();
		target_ptr = (int64_t*)contig_target.data_ptr();

		while (true)
		{
			char_len = strlen(text_examples_[index]);
			if (char_len < 10)
			{
				index = rand() % num_training_examples_;
				continue;
			}

			tokens_ptr[0] = BOS_;
			len = max_context_len_ - 1;
			ret = Encode(tokenizer_, text_examples_[index], &tokens_ptr[1], &len);
			if (ret || (len > char_len))
			{
				index = rand() % num_training_examples_;
			}
			else
			{
				break;
			}
		}

		for (i = 0; i < len; i++)
		{
			target_ptr[i] = tokens_ptr[i + 1];
		}
		target_ptr[len] = EOS_;


		len++; // accomodate BOS_/EOS_
		pad_len = max_context_len_ - len;
		for (i = 0; i < pad_len; i++)
		{
			tokens_ptr[len + i] = PAD_;
			target_ptr[len + i] = PAD_;
		}



		return { tokens, target };
	}
	*/

	torch::optional<size_t> size() const override
	{
		return num_training_examples_;
	}

	uint32_t GetVocabularySize()
	{
		return vocabular_size_;
	}

private:
	torch::Tensor images_;
	enum { temp_buffer_size = 10000 };
	uint64_t num_training_examples_;
	char** text_examples_;
	uint32_t max_context_len_;
	void* tokenizer_;
	uint32_t BOS_;
	uint32_t EOS_;
	uint32_t PAD_;
	uint32_t vocabular_size_;
	uint64_t get_item_retries_;
};
