#Purpose: Manages loading and querying of the local LLM.
# Core Classes & Functions:
#MistralGenerator:
#Sets up BitsAndBytesConfig for NF4 4-bit quantization.
#Loads the Mistral model onto GPU resources (device_map="auto").
#.generate(): Wraps prompts in Mistral's instruction chat template, runs tokenization, invokes self.model.generate() with temperature/top-p controls, and decodes token outputs.
#Includes a rule-based generator fallback (_fallback_generate) for CPU debugging.

from __future__ import annotations
import re

from config import GenerationConfig
import os


class MistralGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.using_fallback_generator = False

        # Allow forcing the lightweight fallback via environment for local/CPU tests. Checked
        # BEFORE loading so we skip the (multi-GB) model download entirely on machines that
        # can't run it — model/tokenizer stay None, and downstream Step E treats that as neutral.
        if os.environ.get("MISTRAL_FALLBACK", "") in ("1", "true", "True"):
            self.using_fallback_generator = True
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization_config = None
            if config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                quantization_config=quantization_config,
            )
        except Exception as exc:
            self.using_fallback_generator = True
            print(
                "WARNING: Mistral could not be loaded in this environment. "
                "Using a simple evidence-based fallback generator for local testing only. "
                f"Original error: {exc}"
            )
        # Allow forcing the lightweight fallback via environment for local tests
        if os.environ.get("MISTRAL_FALLBACK", "") in ("1", "true", "True"):
            self.using_fallback_generator = True

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        # max_new_tokens overrides the configured default for a single call. Stage 1
        # query expansion (llm_query_expansion.py) needs a much shorter budget than
        # Stage 7 answer generation, and reuses this same loaded model.
        if self.using_fallback_generator:
            return self._fallback_generate(prompt)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _fallback_generate(prompt: str) -> str:
        context_match = re.search(
            r"Retrieved Context:\n(?P<context>.*?)\n\nQuestion:\n",
            prompt,
            flags=re.DOTALL,
        )
        question_match = re.search(
            r"Question:\n(?P<question>.*?)\n\nInstructions:",
            prompt,
            flags=re.DOTALL,
        )
        context = context_match.group("context").strip() if context_match else ""
        question = question_match.group("question").strip() if question_match else ""
        sentences = re.split(r"(?<=[.!?])\s+", context)
        evidence_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        evidence = " ".join(evidence_sentences[:3])
        if not evidence:
            evidence = "The retrieved context did not provide enough usable evidence."

        return (
            "Reasoning: This local fallback answer is generated from the retrieved "
            "context only because Mistral could not be loaded in the current Windows "
            "environment. The question is: "
            f"{question} The most relevant retrieved evidence states: {evidence}\n"
            "Conclusion: maybe. The final decision should be produced by Mistral on "
            "HPC for the official experiment."
        )
