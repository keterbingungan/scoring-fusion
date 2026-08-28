from typing import List, Dict, Literal, Optional, Any, Tuple
import re
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams



class LLMJudge:
    """
        Fast RAG that can dynamically retrieve the top-k documents according to the LLM's estimation,
        i.e., the number of retrieved documents is first estimated by the LLM
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 1024,
        USE_VLLM: bool = False,
        GPUID: int = 0
        ) -> None:

        self.max_new_tokens = max_new_tokens
        self.use_vllm = USE_VLLM

        # Initialize the tokenizer and model
        """ We use vLLM to achieve batch inference """
        if self.use_vllm:
            self.model = LLM(model=model_name_or_path, tensor_parallel_size=1)
        # Basic case
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map={"": f"cuda:{GPUID}"}).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map={"": f"cuda:{GPUID}"})
        
        
        

    # Design the evaluation prompt for the LLM
    def get_eval_prompt(self, q: str, gold_ans: str, gen_ans: str) -> List[Dict]:


        prompt = f"""You are an expert evaluation system for a question answering chatbot.

            You are given the following information:
            - the query
            - a generated answer
            - a reference answer

            Your task is to evaluate the correctness of the generated answer.

            ## Query
            {q}

            ## Reference Answer
            {gold_ans}

            ## Generated Answer
            {gen_ans}

            Your response should be formatted as following:
            <judge>True or False</judge>

            If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.

            Please note that the generated answer may contain additional information beyond the reference answer.
        """

        message = [
            {"role": "user", "content": prompt}
        ]

        return message





    # Generate the response that determines whether the generated answer is correct or not
    def generate_response(self, q: str, gold_ans, gen_ans: str) -> bool:
        """
            q: question
            gold_ans: the ground truth answer
            gen_ans: the generated answer
        """
        
        # Prepare the prompt
        user_msg = self.get_eval_prompt(q)
        
        
        # apply_chat_template is the officially recommended way to prepare input
        prompt_text = self.tokenizer.apply_chat_template(
            user_msg,
            tokenize=False,
            add_generation_prompt=True
        )
        # Tokenize
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        
        # Using eos to stop the generation
        eos_word = "</answer>"
        eos_answer_id = self.tokenizer.convert_tokens_to_ids(eos_word)   # stop right after this
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_answer_id        # hard stop
        )
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)
        full_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # answer = full_text.split("Answer:")[-1].strip().split("\n")[0]
        answer = full_text.split("<judge>")[-1].split("</judge>")[0].strip()
        answer = True if answer.lower() == "true" else False

        return answer, full_text
    



    # Judge a batch of questions using the vLLM
    def generate_response_batch(self, qs: List[str], gold_anss: List[str], gen_anss: List[str]) -> List[bool]:
        """
            q: a batch of questions
            docs: list of retrieved documents. None means no retrieval.
            num_responses: the number of responses to generate for each query
            randomness: whether to use randomness in the generation process
            prev_eval: whether it is an pre-evaluation of the given question
        """
        
        # Make sure q is a list
        if not isinstance(qs, list):
            qs = [qs]

        # Prepare the input
        messages = []
        for i, q in enumerate(qs):
            # Prepare the prompt
            user_msg = self.get_eval_prompt(q, gold_anss[i], gen_anss[i])

            # apply_chat_template is the officially recommended way to prepare input
            msg = self.tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True
            )
            # Build a single-turn ChatML prompt
            messages.append(msg)

        
        tmp, top_p, top_k = 0, 1.0, 1
        # parameters
        sampling_params = SamplingParams(
            n=1, # How many responses to generate for each query
            temperature=tmp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            max_tokens=self.max_new_tokens,
            stop=["<|im_end|>"],  # vLLM stops when any token is generated
        )
        # Generation
        outputs = self.model.generate(messages, sampling_params)
        # full_text = [output.outputs[0].text.strip() for output in outputs]
        full_text = [[opt.text.strip() for opt in output.outputs] for output in outputs]
        

        # Extract the answer from the generated text （batch_size * num_responses）
        answer = [[text.split("<judge>")[-1].split("</judge>")[0].strip() for text in texts] for texts in full_text]
        answer = [[True if ans.lower() == "true" else False for ans in ans_list] for ans_list in answer]
        # Get the first answer for each query
        answer = [ans[0] for ans in answer]
        
        return answer, full_text

        