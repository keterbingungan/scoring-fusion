import base64
from typing import List, Dict, Union, Optional
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import torch
import io
from PIL import Image
import math



class MMAgent:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_vllm: bool = True,
        gpu_devices: Optional[List[int]] = None,
        trust_remote_code: bool = True,
        min_pixels = 256*28*28,
        max_pixels = 1280*28*28,
        max_new_tokens: int = 1024
    ):
        """
        Initialize Qwen2.5-VL with multi-image and multi-GPU support.
        
        Args:
            model_name: Model identifier (Hugging Face path).
            use_vllm: Use vLLM for faster text generation (requires manual image handling).
            gpu_devices: List of GPU IDs (e.g., [0, 1] for GPUs 0 and 1). Default: all available.
            trust_remote_code: Allow custom code from Hugging Face (required for Qwen).
        """
        self.use_vllm = use_vllm
        self.model_name = model_name
        self.gpu_devices = gpu_devices
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

        
        # Load processor (required for both backends)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
            )
        
        if use_vllm:
            # vLLM setup with multi-GPU
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=len(gpu_devices) if gpu_devices else torch.cuda.device_count(),
                trust_remote_code=trust_remote_code,
            )
        else:
            # Hugging Face setup with explicit device placement
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self._get_device_map(),
                torch_dtype="auto",
                trust_remote_code=trust_remote_code,
            )


    def _get_device_map(self) -> Union[str, Dict]:
        """Assign model layers to specific GPUs if gpu_devices is provided."""
        if not self.gpu_devices:
            return "auto"
        return {"": f"cuda:{self.gpu_devices[0]}"}  # First GPU as primary


    def _process_images_to_base64(self, image_paths: List[str]) -> str:
        """Convert multiple images to base64 with aspect-ratio-preserving resizing."""
        image_tags = []
        for img_path in image_paths:
            # Open and resize image while preserving aspect ratio
            img = Image.open(img_path).convert("RGB")
            
            # Calculate current and target pixel counts
            current_pixels = img.width * img.height
            
            # Resize only if needed
            if current_pixels < self.min_pixels:
                # Scale up to meet minimum pixels (preserving aspect ratio)
                scale_factor = math.sqrt(self.min_pixels / current_pixels)
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            elif current_pixels > self.max_pixels:
                # Scale down to meet maximum pixels (preserving aspect ratio)
                scale_factor = math.sqrt(self.max_pixels / current_pixels)
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)  # Adjust quality as needed
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_tags.append(f"<image>{base64_str}</image>")
        
        return "\n".join(image_tags)



    def _process_inputs(
        self,
        messages: List[Dict],
    ) -> Union[str, Dict]:
        """
        Process inputs with support for multiple images.
        Returns:
            - vLLM: Single prompt string with embedded images.
            - Hugging Face: Dict of tokenized inputs.
        """
        # Extract images and text from messages
        image_paths = []
        text_parts = []
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "image":
                    image_paths.append(content["image"])
                elif content["type"] == "text":
                    text_parts.append(content["text"])

        # Generate base prompt
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self.use_vllm:
            # For vLLM: Inject all images as base64
            if image_paths:
                image_section = self._process_images_to_base64(image_paths)
                text = f"{text}\n{image_section}"
            return text
        else:
            # For Hugging Face: Use native multimodal processing
            image_inputs, _ = process_vision_info(messages)
            return self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(f"cuda:{self.gpu_devices[0]}" if self.gpu_devices else "cuda")



    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate responses with multi-image and multi-GPU support."""
        """ {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"} """
        inputs = self._process_inputs(messages)
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if self.use_vllm:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            outputs = self.llm.generate(inputs, sampling_params)
            return outputs[0].text.strip()
        else:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            # Trim input tokens and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()



# Example Usage
if __name__ == "__main__":
    # Initialize with 2 GPUs (IDs 0 and 1) and vLLM
    qwen_vllm = MMAgent(
        use_vllm=False,
        gpu_devices=[0, 1],  # Use GPUs 0 and 1
    )

    # Multi-image input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image1.png"},
                {"type": "image", "image": "image2.png"},
                {"type": "text", "text": "Which universities are illustrated in these two pictures? And which is better?"},
            ],
        }
    ]

    response = qwen_vllm.generate(messages, max_new_tokens=512)
    print(response)