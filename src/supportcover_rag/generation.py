from __future__ import annotations

import http.client
import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from supportcover_rag.config import GenerationConfig, PromptingConfig
from supportcover_rag.device import (
    device_name,
    dtype_name,
    is_gpu_device,
    move_batch_to_device,
    pretty_device_name,
    resolve_device,
    resolve_dtype,
)
from supportcover_rag.text import whitespace_token_estimate

LOGGER = logging.getLogger(__name__)

_ASSISTANT_TOKEN_PATTERN = re.compile(r"(?is)^\s*(?:<\|assistant\|>|<assistant>|assistant)\s*[:\-]?\s*")
_FINAL_ANSWER_PREFIX_PATTERN = re.compile(r"(?is)^\s*(?:final answer|answer|response)\s*[:\-]\s*")


@dataclass(slots=True)
class PromptInput:
    question: str
    evidence: str


@dataclass(slots=True)
class GenerationResult:
    text: str
    generated_tokens: int


def build_prompt_messages(prompting: PromptingConfig, prompt_input: PromptInput) -> list[dict[str, str]]:
    evidence_block = prompt_input.evidence.strip() or "No evidence provided."
    user_sections = [
        f"Evidence:\n{evidence_block}",
        f"Question: {prompt_input.question.strip()}",
    ]
    user_instruction = prompting.user_instruction.strip()
    if user_instruction:
        user_sections.append(user_instruction)

    messages: list[dict[str, str]] = []
    system_instruction = prompting.system_instruction.strip()
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": "\n\n".join(user_sections)})
    return messages


def build_completion_prompt(prompting: PromptingConfig, prompt_input: PromptInput) -> str:
    messages = build_prompt_messages(prompting=prompting, prompt_input=prompt_input)
    sections: list[str] = []
    for message in messages:
        sections.append(f"{message['role'].capitalize()}:\n{message['content']}")
    sections.append("Assistant:")
    return "\n\n".join(sections)


def postprocess_generated_text(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = _ASSISTANT_TOKEN_PATTERN.sub("", cleaned).strip()
        updated = _FINAL_ANSWER_PREFIX_PATTERN.sub("", updated).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def render_transformer_prompt(
    tokenizer: object,
    prompting: PromptingConfig,
    prompt_input: PromptInput,
) -> tuple[str, bool]:
    messages = build_prompt_messages(prompting=prompting, prompt_input=prompt_input)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if callable(apply_chat_template) and chat_template:
        rendered = apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(rendered), True
    return build_completion_prompt(prompting=prompting, prompt_input=prompt_input), False


class TokenCounter:
    def count(self, text: str) -> int:
        raise NotImplementedError


class WhitespaceTokenCounter(TokenCounter):
    def count(self, text: str) -> int:
        return whitespace_token_estimate(text)


class TransformersTokenCounter(TokenCounter):
    def __init__(self, model_name_or_path: str, trust_remote_code: bool = False) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    def count(self, text: str) -> int:
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return max(1, len(token_ids))


class BaseGenerator:
    def generate(self, question: str, evidence: str) -> GenerationResult:
        return self.generate_one(PromptInput(question=question, evidence=evidence))

    def generate_one(self, prompt_input: PromptInput) -> GenerationResult:
        return self.generate_batch([prompt_input])[0]

    def generate_batch(self, prompts: list[PromptInput]) -> list[GenerationResult]:
        return [self._generate_one(prompt_input=prompt_input) for prompt_input in prompts]

    def _generate_one(self, prompt_input: PromptInput) -> GenerationResult:
        raise NotImplementedError


class EchoGenerator(BaseGenerator):
    def _generate_one(self, prompt_input: PromptInput) -> GenerationResult:
        text = "insufficient evidence" if not prompt_input.evidence.strip() else prompt_input.evidence.split("\n", 1)[0]
        text = postprocess_generated_text(text)
        return GenerationResult(text=text, generated_tokens=whitespace_token_estimate(text))


class TransformersGenerator(BaseGenerator):
    def __init__(self, config: GenerationConfig, prompting: PromptingConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers and torch are required for generation.") from exc

        self.prompting = prompting
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            fallback_pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback_pad_token is None:
                raise RuntimeError(
                    f"Tokenizer '{config.model_name_or_path}' does not expose a pad_token, eos_token, or unk_token for batched generation."
                )
            self.tokenizer.pad_token = fallback_pad_token
        self.tokenizer.padding_side = "left"
        self.uses_chat_template = bool(getattr(self.tokenizer, "chat_template", None)) and callable(
            getattr(self.tokenizer, "apply_chat_template", None)
        )
        self.device = resolve_device(config.device)
        self.batch_size = max(1, config.batch_size)
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens
        self.do_sample = config.do_sample
        self._torch = torch
        self.model, self.dtype = self._load_model_with_dtype(config, AutoModelForCausalLM)
        self.model.eval()
        if hasattr(self.model, "config") and getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.gpu_acceleration_active = is_gpu_device(self.device)
        LOGGER.info(
            "Generation startup | backend=transformers | model=%s | prompt_format=%s | device=%s (%s) | dtype=%s | batch_size=%d | gpu_acceleration=%s",
            config.model_name_or_path,
            "chat_template" if self.uses_chat_template else "plain_text_fallback",
            device_name(self.device),
            pretty_device_name(self.device),
            dtype_name(self.dtype),
            self.batch_size,
            self.gpu_acceleration_active,
        )

    def _load_model_with_dtype(self, config: GenerationConfig, auto_model_cls: type[object]) -> tuple[object, object]:
        requested_dtype = (config.dtype or "auto").lower()
        preferred_dtype = resolve_dtype(config.dtype, self.device)
        try:
            return self._load_model(config, auto_model_cls, preferred_dtype), preferred_dtype
        except Exception as exc:
            if requested_dtype == "auto" and dtype_name(preferred_dtype) == "float16":
                LOGGER.warning(
                    "float16 model load failed on %s (%s). Falling back to float32.",
                    device_name(self.device),
                    exc,
                )
                return self._load_model(config, auto_model_cls, self._torch.float32), self._torch.float32
            raise RuntimeError(
                f"Failed to load model '{config.model_name_or_path}' on {device_name(self.device)} with dtype {dtype_name(preferred_dtype)}."
            ) from exc

    def _load_model(self, config: GenerationConfig, auto_model_cls: type[object], torch_dtype: object) -> object:
        load_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "dtype": torch_dtype,
        }
        try:
            model = auto_model_cls.from_pretrained(
                config.model_name_or_path,
                **load_kwargs,
            )
        except TypeError:
            model = auto_model_cls.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                torch_dtype=torch_dtype,
            )
        return model.to(self.device)

    def generate_batch(self, prompts: list[PromptInput]) -> list[GenerationResult]:
        if not prompts:
            return []
        rendered_prompts = []
        used_chat_template = False
        for prompt_input in prompts:
            rendered_prompt, prompt_uses_chat_template = render_transformer_prompt(
                tokenizer=self.tokenizer,
                prompting=self.prompting,
                prompt_input=prompt_input,
            )
            rendered_prompts.append(rendered_prompt)
            used_chat_template = used_chat_template or prompt_uses_chat_template
        tokenized_inputs = self.tokenizer(
            rendered_prompts,
            add_special_tokens=not used_chat_template,
            padding=True,
            return_tensors="pt",
        )
        prompt_width = int(tokenized_inputs["input_ids"].shape[1])
        inputs = move_batch_to_device(tokenized_inputs, self.device)
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if device_name(self.device) != "cpu":
            output = output.to("cpu")

        results: list[GenerationResult] = []
        for sequence in output:
            generated_ids = sequence[prompt_width:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            cleaned_text = postprocess_generated_text(text)
            results.append(GenerationResult(text=cleaned_text, generated_tokens=int(generated_ids.numel())))
        return results


class OllamaError(RuntimeError):
    pass


class OllamaHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported Ollama base URL '{base_url}'. Expected http:// or https://.")
        if not parsed.hostname:
            raise ValueError(f"Could not parse Ollama base URL '{base_url}'.")

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.use_tls = parsed.scheme == "https"
        path_prefix = parsed.path.rstrip("/")
        self.api_prefix = path_prefix if path_prefix.endswith("/api") else f"{path_prefix}/api" if path_prefix else "/api"
        self._connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _create_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        connection_cls = http.client.HTTPSConnection if self.use_tls else http.client.HTTPConnection
        return connection_cls(self.host, self.port, timeout=self.timeout_seconds)

    def _request_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for _ in range(2):
            if self._connection is None:
                self._connection = self._create_connection()
            try:
                self._connection.request("POST", f"{self.api_prefix}{path}", body=body, headers=headers)
                response = self._connection.getresponse()
                raw_body = response.read()
                status = response.status
                if status >= 400:
                    raise OllamaError(self._extract_error(raw_body) or f"Ollama returned HTTP {status}.")
                if not raw_body:
                    return {}
                return json.loads(raw_body.decode("utf-8"))
            except OllamaError:
                raise
            except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError) as exc:
                last_error = exc
                self.close()

        raise OllamaError(
            f"Failed to reach Ollama at {self.base_url}. Ensure the Ollama server is running and reachable."
        ) from last_error

    @staticmethod
    def _extract_error(raw_body: bytes) -> str:
        text = raw_body.decode("utf-8", errors="replace").strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        error = payload.get("error")
        if isinstance(error, str):
            return error
        return text

    def post_chat(self, payload: dict[str, object]) -> dict[str, object]:
        return self._request_json("/chat", payload)


class OllamaGenerator(BaseGenerator):
    def __init__(self, config: GenerationConfig, prompting: PromptingConfig) -> None:
        self.prompting = prompting
        self.model_name = config.model_name_or_path
        self.base_url = config.base_url.rstrip("/")
        self.timeout_seconds = float(config.timeout_seconds)
        self.think = bool(config.think)
        self.stream = bool(config.stream)
        self.batch_size = max(1, config.batch_size)
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens
        self.do_sample = config.do_sample
        self.client = OllamaHttpClient(base_url=self.base_url, timeout_seconds=self.timeout_seconds)
        LOGGER.info(
            "Generation startup | backend=ollama | model=%s | base_url=%s | think=%s | stream=%s | batch_size=%d",
            self.model_name,
            self.base_url,
            str(self.think).lower(),
            str(self.stream).lower(),
            self.batch_size,
        )

    def _build_request_payload(self, prompt_input: PromptInput) -> dict[str, object]:
        return {
            "model": self.model_name,
            "messages": build_prompt_messages(self.prompting, prompt_input),
            "stream": self.stream,
            "think": self.think,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": self.temperature if self.do_sample else 0.0,
            },
        }

    def _generate_one(self, prompt_input: PromptInput) -> GenerationResult:
        payload = self._build_request_payload(prompt_input)
        LOGGER.debug(
            "Ollama request | model=%s | think=%s | stream=%s | max_new_tokens=%d",
            self.model_name,
            str(self.think).lower(),
            str(self.stream).lower(),
            self.max_new_tokens,
        )
        try:
            response = self.client.post_chat(payload)
        except OllamaError as exc:
            raise RuntimeError(
                f"Ollama generation failed for model '{self.model_name}' via {self.base_url}: {exc}"
            ) from exc

        message = response.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(
                f"Ollama returned an unexpected response for model '{self.model_name}': missing 'message' payload."
            )

        raw_text = message.get("content", "")
        if not isinstance(raw_text, str):
            raise RuntimeError(
                f"Ollama returned an unexpected response for model '{self.model_name}': message.content was not a string."
            )

        cleaned_text = postprocess_generated_text(raw_text)
        generated_tokens = response.get("eval_count")
        if not isinstance(generated_tokens, int):
            generated_tokens = whitespace_token_estimate(cleaned_text)
        return GenerationResult(text=cleaned_text, generated_tokens=max(0, generated_tokens))


def build_generator(config: GenerationConfig, prompting: PromptingConfig) -> BaseGenerator:
    backend = config.backend.lower()
    if backend == "echo":
        LOGGER.warning("Using echo backend. This is for smoke tests only, not for real experiments.")
        return EchoGenerator()
    if backend == "transformers":
        return TransformersGenerator(config=config, prompting=prompting)
    if backend == "ollama":
        return OllamaGenerator(config=config, prompting=prompting)
    raise ValueError(f"Unsupported generation backend: {config.backend}")


def build_token_counter(config: GenerationConfig) -> TokenCounter:
    if config.backend.lower() == "transformers":
        try:
            return TransformersTokenCounter(
                model_name_or_path=config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - graceful fallback for light environments
            LOGGER.warning("Falling back to whitespace token counter because tokenizer loading failed: %s", exc)
    return WhitespaceTokenCounter()
