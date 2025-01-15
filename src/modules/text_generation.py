import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import threading
import queue

class TextGenerator:
    def __init__(self):
        self.model_dir = "EleutherAI/gpt-neo-2.7B"  # Directory where the model is saved
        self.model = None
        self.tokenizer = None
        self.generating_text = False
        self.result_cache = {}
        self.result_cache_lock = threading.Lock()
        self.batch_queue = queue.Queue()

    async def start_text_generation(self):
        await self._lazy_load_pipeline()
        self.generating_text = True
        # Start batch processing thread
        threading.Thread(target=self._batch_processing_thread).start()

    async def stop_text_generation(self):
        self.generating_text = False

    async def generate_text(self, prompts):
        if not self.generating_text:
            return "Text generation is not active."

        if isinstance(prompts, str):
            prompts = [prompts]

        results = []

        for prompt in prompts:
            if prompt in self.result_cache:
                results.append(self.result_cache[prompt])
            else:
                self.batch_queue.put(prompt)

        return await self._batch_generate_text(results)

    async def _lazy_load_pipeline(self):
        # Preload the model and tokenizer
        self.model = GPTNeoForCausalLM.from_pretrained(self.model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)

        # Adjusting max_length parameter
        self.tokenizer.model_max_length = 1024

    def _batch_processing_thread(self):
        while True:
            if not self.generating_text:
                break
            batch_prompts = []
            while not self.batch_queue.empty():
                batch_prompts.append(self.batch_queue.get())
            if batch_prompts:
                results = self._batch_generate_text(batch_prompts)
                with self.result_cache_lock:
                    for prompt, result in zip(batch_prompts, results):
                        self.result_cache[prompt] = result

    async def _batch_generate_text(self, prompts):
        responses = []

        for i in range(0, len(prompts), 4):
            batch_prompts = prompts[i:i + 4]
            batch_responses = await asyncio.gather(*[self._generate_single_text(prompt) for prompt in batch_prompts])
            responses.extend(batch_responses)

        return responses

    async def _generate_single_text(self, prompt):
        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(input_ids)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text
