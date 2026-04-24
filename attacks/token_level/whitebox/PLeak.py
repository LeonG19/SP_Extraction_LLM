import numpy as np
import warnings
import torch
import re
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class PLeakAttack:
    def __init__(
        self,
        model="llama3",
        optim_token_length=20,
        num_trigger=5,
        input_trigger="",
        use_english_vocab=False,
        top_k=50,
        temperature=0.5,
        learning_rate=0.1,
        num_iterations=100,
    ):
        """
        PLeak: Prompt Leaking Attacks against Large Language Models

        Args:
            model: Target model name (llama3, llama, etc.)
            optim_token_length: Number of tokens to optimize
            num_trigger: Number of adversarial prompts to generate
            input_trigger: Initial seed prompt
            use_english_vocab: Whether to restrict to English tokens
            top_k: Number of top candidates for token replacement
            temperature: Temperature for candidate selection
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations per trigger
        """
        assert optim_token_length > 0 and temperature > 0, "Invalid parameter!"

        model_to_name = dict(
            zip(
                [
                    "gptj",
                    "opt",
                    "llama",
                    "llama-70b",
                    "llama-chat",
                    "llama3",
                    "llama3-80b",
                    "falcon",
                    "vicuna",
                    "mixtral",
                ],
                [
                    "EleutherAI/gpt-j-6b",
                    "facebook/opt-6.7B",
                    "meta-llama/Llama-2-7b-hf",
                    "meta-llama/Llama-2-70b-chat-hf",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "tiiuae/falcon-7b",
                    "lmsys/vicuna-7b-v1.5",
                    'mistralai/Mistral-7B-Instruct-v0.2',
                ],
            )
        )
        model_name = model_to_name.get(model)
        if not model_name:
            raise ValueError(f"Unknown model: {model}")

        self.model_name = model
        print(f"\n{'='*60}")
        print(f"Loading PLeak with model: {model}")
        print(f"HuggingFace model: {model_name}")
        print(f"{'='*60}\n")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=quant_config
        ).eval()
        print(f"Model loaded successfully on device: {self.model.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        # Set up chat template for llama models
        if model == "llama":
            chat_template = r"""{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
                            {% elif message['role'] == 'system' %}\{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
                            {% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"""
            self.tokenizer.chat_template = chat_template

        if model in ["llama3", "llama3-80b"]:
            self.tokenizer.pad_token_id = 128001  # Use eos_token as pad
        else:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim_token_length = optim_token_length
        self.input_trigger = input_trigger
        self.use_english_vocab = use_english_vocab
        self.top_k = top_k
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_trigger = num_trigger

        self.trigger = None
        self.generated_triggers = []
        self.repetition_record = [{} for _ in range(optim_token_length)]

        # Set vocabulary size
        model_to_vocab = dict(
            zip(["gptj", "opt", "falcon", "llama3", "llama3-80b"],
                [50400, 50272, 65024, 128256, 128256])
        )
        self.vocab_size = model_to_vocab.get(model, 32000)

    def init_triggers(self, trigger_token_length):
        """Initialize adversarial trigger tokens"""
        input_tokens = self.tokenizer.encode(self.input_trigger, add_special_tokens=False)
        input_len = len(input_tokens)

        if input_len > trigger_token_length:
            warnings.warn(
                "The initial token is too long and may lead to inadequate optimization",
                UserWarning,
            )
            return np.asarray(input_tokens[:trigger_token_length])
        else:
            trigger = np.array(input_tokens, dtype=int)
            while input_len < trigger_token_length:
                t = np.random.randint(self.vocab_size)
                while (
                    re.search(r"[^a-zA-Z0-9s\s]", self.tokenizer.decode(t))
                    and self.use_english_vocab
                ):
                    t = np.random.randint(self.vocab_size)
                trigger = np.append(trigger, t)
                input_len += 1
            return trigger

    def compute_prompt_extraction_loss(self, system_prompt, trigger):
        """
        Compute loss for prompt extraction.
        Lower loss means the model is more likely to output the system prompt
        when given the adversarial trigger.
        """
        # Create conversation where we ask the model to reveal the system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.tokenizer.decode(trigger, skip_special_tokens=True)},
            {"role": "assistant", "content": system_prompt},
        ]

        # Encode with chat template
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs["input_ids"]

        # Create labels - only compute loss on the assistant's response (the leaked prompt)
        non_label = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.tokenizer.decode(trigger, skip_special_tokens=True)},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        non_label_ids = self.tokenizer(
            non_label,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:, :non_label_ids] = -100  # Ignore tokens before assistant response

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss

        return loss

    def get_trigger_gradients(self, system_prompt, trigger):
        """Compute gradients w.r.t. trigger tokens"""
        self.model.zero_grad()
        loss = self.compute_prompt_extraction_loss(system_prompt, trigger)
        loss.backward()

        # Get gradients w.r.t. embeddings
        embed_layer = self.model.model.embed_tokens
        if embed_layer.weight.grad is not None:
            grads = embed_layer.weight.grad.clone()
        else:
            grads = None

        return loss.item(), grads

    def hotflip_candidates(self, gradients):
        """Find top-k token candidates using gradient-based method"""
        # Get embedding matrix
        embed_layer = self.model.model.embed_tokens
        embedding_matrix = embed_layer.weight.detach()

        # Compute top-k candidates for each token position
        # Process one token at a time to save memory
        top_indices_list = []

        for token_idx in range(gradients.shape[0]):
            grad = gradients[token_idx]  # [embedding_dim]

            # Compute dot product with all embedding vectors
            # More memory efficient: compute scores in chunks
            scores = torch.matmul(grad, embedding_matrix.t())  # [vocab_size]

            # Get top-k candidates
            _, top_k_ids = torch.topk(scores, self.top_k, dim=0)
            top_indices_list.append(top_k_ids.cpu().numpy())

        return np.array(top_indices_list)

    def optimize_trigger(self, system_prompts):
        """Optimize trigger tokens to maximize prompt extraction"""
        # Sample a subset of prompts to speed up training
        sample_size = min(10, len(system_prompts))
        sampled_prompts = np.random.choice(len(system_prompts), sample_size, replace=False)
        sampled_prompts = [system_prompts[i] for i in sampled_prompts]

        best_loss = float('inf')
        best_trigger = deepcopy(self.trigger)

        for iteration in range(self.num_iterations):
            # Compute loss and gradients for current trigger
            total_loss = 0
            all_gradients = None

            with torch.set_grad_enabled(True):
                for system_prompt in sampled_prompts:
                    loss, grads = self.get_trigger_gradients(system_prompt, self.trigger)
                    total_loss += loss.item() if hasattr(loss, 'item') else loss
                    if all_gradients is None:
                        all_gradients = grads.clone()
                    else:
                        all_gradients = all_gradients + grads

            total_loss /= len(sampled_prompts)

            if iteration % 10 == 0:
                trigger_text = self.tokenizer.decode(self.trigger, skip_special_tokens=True)[:50]
                print(f"Iteration {iteration}, Loss: {total_loss:.4f}, Trigger: {trigger_text}...")

            if total_loss < best_loss:
                best_loss = total_loss
                best_trigger = deepcopy(self.trigger)

            # Get candidate tokens
            if all_gradients is not None:
                all_gradients = all_gradients / len(sampled_prompts)
                candidates = self.hotflip_candidates(all_gradients)

                # Greedily replace one token that improves loss the most
                best_replacement_pos = None
                best_replacement_token = None
                best_replacement_loss = total_loss

                for pos in range(len(self.trigger)):
                    # Only try top 5 candidates per position to save time
                    for cand_idx in range(min(5, len(candidates[pos]))):
                        cand_token = candidates[pos][cand_idx]

                        # Skip if special character and use_english_vocab is True
                        if re.search(r"[^a-zA-Z0-9s\s]", self.tokenizer.decode(cand_token)) and self.use_english_vocab:
                            continue

                        # Test this candidate
                        test_trigger = deepcopy(self.trigger)
                        test_trigger[pos] = cand_token

                        # Compute loss with candidate
                        test_loss = 0
                        with torch.no_grad():
                            for system_prompt in sampled_prompts:
                                try:
                                    loss, _ = self.get_trigger_gradients(system_prompt, test_trigger)
                                    test_loss += loss.item() if hasattr(loss, 'item') else loss
                                except:
                                    continue

                        test_loss /= len(sampled_prompts)

                        # Track best replacement
                        if test_loss < best_replacement_loss:
                            best_replacement_loss = test_loss
                            best_replacement_pos = pos
                            best_replacement_token = cand_token

                # Apply best replacement if found
                if best_replacement_pos is not None:
                    self.trigger[best_replacement_pos] = best_replacement_token
                    total_loss = best_replacement_loss

        self.trigger = deepcopy(best_trigger)
        return best_loss

    def train(self, system_prompts):
        """
        Generate adversarial triggers to extract system prompts.

        Args:
            system_prompts: List of system prompts to extract
        """
        print(f"\n{'='*60}")
        print(f"Starting PLeak Attack")
        print(f"{'='*60}")
        print(f"Target model: {self.model_name}")
        print(f"Number of prompts to extract: {len(system_prompts)}")
        print(f"Trigger length: {self.optim_token_length}")
        print(f"{'='*60}\n")

        for trigger_idx in range(self.num_trigger):
            print(f"\nGenerating trigger {trigger_idx + 1}/{self.num_trigger}...")

            # Initialize trigger
            self.trigger = self.init_triggers(self.optim_token_length)
            print(f"Initial trigger: {self.tokenizer.decode(self.trigger, skip_special_tokens=True)}")

            # Optimize trigger
            final_loss = self.optimize_trigger(system_prompts)

            # Record result
            trigger_text = self.tokenizer.decode(self.trigger, skip_special_tokens=True)
            print(f"Final trigger: {trigger_text}")
            print(f"Final loss: {final_loss:.4f}")

            self.generated_triggers.append({
                "trigger": trigger_text,
                "loss": final_loss,
                "token_ids": self.trigger.copy()
            })

            # Update repetition record
            for pos in range(len(self.trigger)):
                token_id = self.trigger[pos]
                if token_id in self.repetition_record[pos]:
                    self.repetition_record[pos][token_id] += 1
                else:
                    self.repetition_record[pos][token_id] = 1

        print(f"\n{'='*60}")
        print(f"PLeak attack completed!")
        print(f"Total triggers generated: {len(self.generated_triggers)}")
        print(f"Best loss: {min(t['loss'] for t in self.generated_triggers):.4f}")
        print(f"{'='*60}\n")

        return self.generated_triggers
