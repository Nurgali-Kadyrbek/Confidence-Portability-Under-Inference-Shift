"""GPU-only vLLM backend with CPU/swap offload structurally disabled."""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cpis.config import ModelSpec, SamplingSpec
from cpis.inference.base import (
    ChatInferenceRequest,
    InferenceRequest,
    InferenceResponse,
)


class VllmBackend:
    def __init__(
        self,
        model: ModelSpec,
        expected_version: str,
        download_dir: str,
        engine_seed: int,
        use_flashinfer_sampler: bool,
        tokenizer_mode: str | None = None,
        config_format: str | None = None,
        load_format: str | None = None,
        worker_multiproc_method: str | None = None,
    ) -> None:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
            "1" if use_flashinfer_sampler else "0"
        )
        if worker_multiproc_method is not None:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = worker_multiproc_method
        try:
            installed = version("vllm")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "install the pinned CUDA 12.9 GPU environment from requirements/gpu-cu129.lock"
            ) from exc
        if installed != expected_version:
            raise RuntimeError(
                f"vLLM version mismatch: config={expected_version}, installed={installed}"
            )
        try:
            import torch
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "install the pinned CUDA 12.9 GPU environment from requirements/gpu-cu129.lock"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM execution requires CUDA; CPU execution is prohibited")

        self._version = installed
        self._model_spec = model
        self._download_dir = download_dir
        loader_options = {
            key: value
            for key, value in {
                "tokenizer_mode": tokenizer_mode,
                "config_format": config_format,
                "load_format": load_format,
            }.items()
            if value is not None
        }
        self._llm = LLM(
            model=model.model_id,
            revision=model.revision,
            tokenizer=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            dtype=model.dtype,
            tensor_parallel_size=model.tensor_parallel_size,
            max_model_len=model.max_model_len,
            gpu_memory_utilization=model.gpu_memory_utilization,
            trust_remote_code=model.trust_remote_code,
            language_model_only=model.language_model_only,
            enforce_eager=model.enforce_eager,
            seed=engine_seed,
            download_dir=download_dir,
            cpu_offload_gb=0,
            cpu_offload_params=set(),
            offload_group_size=0,
            **loader_options,
        )
        self._verify_chat_template()

    @property
    def version(self) -> str:
        return self._version

    def _verify_chat_template(self) -> None:
        if self._model_spec.prompt_format == "plain":
            return
        tokenizer = self._llm.get_tokenizer()
        if getattr(tokenizer, "IS_MISTRAL_TOKENIZER", False):
            from huggingface_hub import hf_hub_download

            artifact = Path(
                hf_hub_download(
                    repo_id=self._model_spec.tokenizer_id,
                    filename="chat_template.jinja",
                    revision=self._model_spec.tokenizer_revision,
                    cache_dir=self._download_dir,
                    local_files_only=True,
                )
            )
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual != self._model_spec.chat_template_sha256:
                raise RuntimeError(
                    "Mistral chat-template artifact SHA-256 differs from the "
                    f"pinned configuration: expected "
                    f"{self._model_spec.chat_template_sha256}, got {actual}"
                )
            return
        template = tokenizer.chat_template
        if isinstance(template, dict):
            template = template.get(self._model_spec.chat_template_id or "")
        if not isinstance(template, str):
            raise RuntimeError("tokenizer has no concrete chat template")
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if actual != self._model_spec.chat_template_sha256:
            raise RuntimeError(
                "chat template SHA-256 differs from the pinned configuration: "
                f"expected {self._model_spec.chat_template_sha256}, got {actual}"
            )

    def _render(
        self, prompt: str, chat_template_kwargs: dict | None = None
    ) -> str | dict[str, Any]:
        if self._model_spec.prompt_format == "plain":
            return prompt
        tokenizer = self._llm.get_tokenizer()
        kwargs = (
            self._model_spec.chat_template_kwargs
            if chat_template_kwargs is None
            else chat_template_kwargs
        )
        messages = []
        if self._model_spec.system_prompt is not None:
            messages.append(
                {"role": "system", "content": self._model_spec.system_prompt}
            )
        messages.append({"role": "user", "content": prompt})
        return self._render_messages(messages, None, kwargs)

    def _render_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict | None = None,
    ) -> str | dict[str, Any]:
        """Render native multi-turn/tool history without retokenizing Mistral."""

        tokenizer = self._llm.get_tokenizer()
        kwargs = (
            self._model_spec.chat_template_kwargs
            if chat_template_kwargs is None
            else chat_template_kwargs
        )
        tool_kwargs = {} if tools is None else {"tools": tools}
        if getattr(tokenizer, "IS_MISTRAL_TOKENIZER", False):
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **tool_kwargs,
                **kwargs,
            )
            if not isinstance(token_ids, list) or not all(
                isinstance(token_id, int) for token_id in token_ids
            ):
                raise RuntimeError("Mistral renderer did not return token IDs")
            # Native Mistral templates must be passed as tokens. Rendering to
            # text and asking vLLM to tokenize again can duplicate BOS/framing.
            return {"prompt_token_ids": token_ids}
        template = tokenizer.chat_template
        if isinstance(template, dict):
            template = template.get(self._model_spec.chat_template_id or "")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
            **tool_kwargs,
            **kwargs,
        )

    @staticmethod
    def _sampling_params(sampling: SamplingSpec, seed: int):
        from vllm import SamplingParams

        return SamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k is not None else 0,
            min_p=sampling.min_p if sampling.min_p is not None else 0.0,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            repetition_penalty=sampling.repetition_penalty,
            max_tokens=sampling.max_tokens,
            min_tokens=sampling.min_tokens,
            ignore_eos=sampling.ignore_eos,
            detokenize=sampling.detokenize,
            skip_special_tokens=sampling.skip_special_tokens,
            spaces_between_special_tokens=sampling.spaces_between_special_tokens,
            include_stop_str_in_output=sampling.include_stop_str_in_output,
            stop=list(sampling.stop) or None,
            stop_token_ids=None,
            bad_words=None,
            logprobs=None,
            prompt_logprobs=None,
            structured_outputs=None,
            logit_bias=None,
            allowed_token_ids=None,
            thinking_token_budget=sampling.thinking_token_budget,
            seed=seed,
        )

    def _generate_rendered(
        self,
        request_ids: list[str],
        prompts: list[Any],
        parameters: list[Any],
    ) -> list[InferenceResponse]:
        outputs = self._llm.generate(prompts, parameters, use_tqdm=False)
        if len(outputs) != len(request_ids):
            raise RuntimeError("vLLM returned the wrong number of request outputs")
        responses: list[InferenceResponse] = []
        for request_id, output in zip(request_ids, outputs, strict=True):
            candidate = output.outputs[0]
            responses.append(
                InferenceResponse(
                    request_id=request_id,
                    text=candidate.text,
                    finish_reason=str(candidate.finish_reason),
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(candidate.token_ids),
                )
            )
        return responses

    def generate(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        prompts: list[Any] = []
        parameters: list[Any] = []
        for request in requests:
            sampling = getattr(request.condition, request.stage)
            parameters.append(self._sampling_params(sampling, request.seed))
            prompts.append(
                self._render(request.prompt, request.chat_template_kwargs)
            )
        return self._generate_rendered(
            [request.request_id for request in requests], prompts, parameters
        )

    def generate_chat(
        self, requests: list[ChatInferenceRequest]
    ) -> list[InferenceResponse]:
        """Generate native tool-use turns from complete structured histories."""

        prompts: list[Any] = []
        parameters: list[Any] = []
        for request in requests:
            prompts.append(
                self._render_messages(
                    list(request.messages),
                    list(request.tools),
                    request.chat_template_kwargs,
                )
            )
            parameters.append(
                self._sampling_params(request.sampling, request.seed)
            )
        return self._generate_rendered(
            [request.request_id for request in requests], prompts, parameters
        )
