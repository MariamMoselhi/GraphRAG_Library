# graph/llm_backend.py
"""
Shared Groq inference helper used by both NodeExtractor
and RelationshipExtractor.

Calls the Groq API (https://api.groq.com/openai/v1/chat/completions)
which is OpenAI-compatible. No local model or GPU is required.

Example
-------
    from graph.llm_backend import LLMBackend

    llm = LLMBackend(api_key="sk-or-...")

    # Single call
    response = llm.generate("Your prompt here")

    # Batched call — combines multiple chunks into one API call
    responses = llm.generate_batch(["prompt 1", "prompt 2", "prompt 3"])

Environment variable shortcut
------------------------------
    Set GROQ_API_KEY in your environment and omit the api_key argument:

        export GROQ_API_KEY="gsk_..."
        llm = LLMBackend()
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional, List

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


_DEFAULT_MODEL       = "llama-3.3-70b-versatile"
_GROQ_URL            = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MAX_TOKENS  = 1024
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_BATCH_SIZE  = 1          # chunks per API call
_RETRY_SLEEP_SECONDS = 15.0
_BATCH_SEPARATOR     = "\n\n---CHUNK_SEPARATOR---\n\n"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMBackend:
    """
    Thin wrapper around the Groq chat-completions API.

    Sends a single user message and returns the assistant's reply text.
    No local model weights, torch, or transformers are required.

    Args
    ----
    api_key     : Groq API key. Falls back to GROQ_API_KEY environment
                  variable if omitted.
    model       : Groq model string.
    max_tokens  : Maximum tokens to generate (default 1024).
    temperature : Sampling temperature -- 0.0 = greedy (default 0.0).
    batch_size  : How many chunks to combine per API call in
                  generate_batch() (default 1).
    site_url    : Unused for Groq; accepted for backward compatibility.
    site_name   : Unused for Groq; accepted for backward compatibility.
    """

    def __init__(
        self,
        api_key    : Optional[str] = None,
        model      : Optional[str] = None,
        max_tokens : int           = _DEFAULT_MAX_TOKENS,
        temperature: float         = _DEFAULT_TEMPERATURE,
        batch_size : int           = _DEFAULT_BATCH_SIZE,
        site_url   : str           = "",
        site_name  : str           = "",
    ):
        if load_dotenv is not None:
            # Load .env from project root/current working directory when available.
            load_dotenv()

        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Pass api_key=... or set GROQ_API_KEY."
            )
        self.model       = model or _DEFAULT_MODEL
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.batch_size  = batch_size
        self.site_url    = site_url
        self.site_name   = site_name
        self.api_url     = _GROQ_URL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Send a single prompt and return the assistant reply.
        Retries up to 3 times on transient provider/network failures
        (e.g. 429/5xx) with fixed backoff.
        """
        return self._call_api(prompt)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Process a list of prompts while minimising API calls by combining
        up to ``self.batch_size`` prompts into a single request.

        Each prompt is wrapped with a numbered header and separated by
        ``---CHUNK_SEPARATOR---`` so the model knows they are distinct
        items. The combined response is split back on the same separator
        and returned as a list aligned 1-to-1 with the input prompts.

        If the model's response cannot be split cleanly (e.g. it omits a
        separator), the raw response is returned for that batch so nothing
        is silently lost.

        Args
        ----
        prompts : list of prompt strings, one per chunk.

        Returns
        -------
        list of response strings, same length as ``prompts``.
        """
        results: List[str] = []
        batch_size = max(1, self.batch_size)

        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start : batch_start + batch_size]

            if len(batch) == 1:
                # No need to wrap a single prompt
                results.append(self._call_api(batch[0]))
                continue

            # Build a combined prompt
            combined_prompt = self._build_batch_prompt(batch)

            # Allow more tokens for a larger batch
            effective_max_tokens = self.max_tokens * len(batch)
            combined_response = self._call_api(
                combined_prompt,
                max_tokens=effective_max_tokens,
            )

            # Split the response back into per-chunk answers
            split = self._split_batch_response(combined_response, expected=len(batch))
            results.extend(split)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_batch_prompt(self, prompts: List[str]) -> str:
        """
        Combine multiple prompts into one, with numbered headers and
        separator instructions so the model returns aligned answers.
        """
        header = (
            f"You will receive {len(prompts)} separate text chunks below. "
            f"Process each one independently and return your answer for each "
            f"chunk separated EXACTLY by the token: ---CHUNK_SEPARATOR---\n"
            f"Do NOT add any text before chunk 1 or after the last chunk answer.\n\n"
        )
        parts = []
        for i, prompt in enumerate(prompts, 1):
            parts.append(f"### CHUNK {i} ###\n{prompt}")

        return header + _BATCH_SEPARATOR.join(parts)

    def _split_batch_response(self, response: str, expected: int) -> List[str]:
        """
        Split a combined response on ``---CHUNK_SEPARATOR---``.
        If the count doesn't match, pad or trim gracefully.
        """
        parts = [p.strip() for p in response.split("---CHUNK_SEPARATOR---")]

        if len(parts) == expected:
            return parts

        # Too few splits — pad missing entries with empty string
        while len(parts) < expected:
            parts.append("")

        # Too many splits — merge the extras into the last entry
        if len(parts) > expected:
            parts = parts[:expected - 1] + ["---CHUNK_SEPARATOR---".join(parts[expected - 1:])]

        return parts

    def _call_api(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Core HTTP call to Groq with retry on transient failures.
        """
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        payload = {
            "model"      : self.model,
            "max_tokens" : effective_max_tokens,
            "temperature": self.temperature,
            "messages"   : [{"role": "user", "content": prompt}],
        }

        headers = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept"       : "application/json",
            "User-Agent"   : "GraphRAG_Library/1.0",
        }

        data = json.dumps(payload).encode("utf-8")

        max_retries = 3
        for attempt in range(max_retries):
            req = urllib.request.Request(
                self.api_url, data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    body = json.loads(resp.read().decode("utf-8"))

                if isinstance(body, dict) and "error" in body:
                    error_obj = body.get("error") or {}
                    error_code = int(error_obj.get("code", 0) or 0)
                    error_msg = error_obj.get("message", "Unknown Groq error")
                    if error_code in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                        wait = _RETRY_SLEEP_SECONDS
                        print(f"  Groq transient error ({error_code}), retrying in {wait}s "
                              f"(attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Groq API error {error_code}: {error_msg}")

                try:
                    return body["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"Unexpected Groq response format: {body}"
                    ) from exc

            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    wait = _RETRY_SLEEP_SECONDS
                    print(f"  HTTP {exc.code}, retrying in {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 403 and "1010" in error_body:
                    raise RuntimeError(
                        "Groq API error 403 (Cloudflare 1010: Access denied). "
                        "This usually indicates an IP/network/region restriction or firewall policy. "
                        "Try a different network, disable VPN/proxy, and verify the API key/project settings in Groq dashboard."
                    ) from exc
                raise RuntimeError(
                    f"Groq API error {exc.code}: {error_body}"
                ) from exc

            except urllib.error.URLError as exc:
                if attempt < max_retries - 1:
                    wait = _RETRY_SLEEP_SECONDS
                    print(f"  Network error, retrying in {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Network error calling Groq: {exc.reason}"
                ) from exc

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMBackend(model={self.model!r}, "
            f"max_tokens={self.max_tokens}, "
            f"batch_size={self.batch_size}, "
            f"temperature={self.temperature})"
        )