"""Deterministic smoke backend. Never permitted for scientific runs."""

from __future__ import annotations

from cpis.inference.base import InferenceRequest, InferenceResponse


class MockBackend:
    def __init__(
        self, answer_response: str, confidence_response: str, version: str = "builtin-1"
    ) -> None:
        self.responses = {
            "answer": answer_response,
            "confidence": confidence_response,
        }
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def generate(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        return [
            InferenceResponse(
                request_id=request.request_id,
                text=self.responses[request.stage],
                finish_reason="mock_complete",
            )
            for request in requests
        ]
