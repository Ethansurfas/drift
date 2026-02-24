"""Drift AI module — ask, classify, embed, see, predict, enrich, score.

All AI primitives dispatch through _call_model(), which routes to
Anthropic or OpenAI based on drift.config.
"""

import json
import os
import base64

from drift_runtime.config import get_config
from drift_runtime.types import (
    ConfidentValue,
    schema_to_json_description,
    parse_ai_response_to_schema,
    _to_drift_dict,
)
from drift_runtime.exceptions import DriftAIError


class DriftAI:
    """AI inference engine for Drift programs."""

    def _call_model(self, messages: list[dict], model: str | None = None) -> str:
        """Call the configured AI provider. Returns the text response."""
        config = get_config()
        provider = config["ai"]["provider"]
        model = model or config["ai"]["default_model"]
        timeout = config["ai"]["timeout"]

        try:
            if provider == "anthropic":
                return self._call_anthropic(messages, model, timeout)
            elif provider == "openai":
                return self._call_openai(messages, model, timeout)
            else:
                raise DriftAIError(f"Unknown AI provider: {provider}")
        except DriftAIError:
            raise
        except Exception as e:
            raise DriftAIError(f"AI call failed: {e}") from e

    def _call_anthropic(self, messages: list[dict], model: str, timeout: int) -> str:
        """Call Anthropic's API."""
        import anthropic

        client = anthropic.Anthropic()

        system_msg = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": user_messages,
            "timeout": timeout,
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _call_openai(self, messages: list[dict], model: str, timeout: int) -> str:
        """Call OpenAI's API."""
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )
        return response.choices[0].message.content

    def ask(self, prompt: str, schema=None, context: dict = None) -> object:
        """Ask the AI a question. Optionally parse response into a schema."""
        user_content = prompt
        if context:
            user_content += f"\n\nContext:\n{json.dumps(context, indent=2)}"

        messages = [{"role": "user", "content": user_content}]

        if schema:
            desc = schema_to_json_description(schema)
            messages.insert(0, {
                "role": "system",
                "content": f"Respond with valid JSON matching this schema:\n{desc}\n\nReturn ONLY the JSON object, no other text.",
            })

        response = self._call_model(messages)

        if schema:
            return parse_ai_response_to_schema(response, schema)
        return response

    def classify(self, input: str, categories: list[str]) -> str:
        """Classify input into one of the provided categories."""
        prompt = (
            f"Classify the following text into exactly one of these categories: "
            f"{categories}\n\nText: {input}\n\n"
            f"Respond with only the category name, nothing else."
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._call_model(messages).strip()

        if response in categories:
            return response

        # Retry once if response didn't match
        retry_prompt = (
            f"Your response '{response}' was not one of the valid categories. "
            f"Choose exactly one of: {categories}\n\nRespond with only the category name."
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": retry_prompt})
        return self._call_model(messages).strip()

    def embed(self, input: str) -> list[float]:
        """Generate an embedding vector for the input text."""
        config = get_config()
        provider = config["ai"]["provider"]

        if provider == "openai":
            import openai
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=input,
            )
            return response.data[0].embedding

        # For Anthropic, use a prompt-based approach
        messages = [{"role": "user", "content": (
            f"Generate a numerical embedding vector for this text as a JSON array of floats. "
            f"Return ONLY the JSON array.\n\nText: {input}"
        )}]
        response = self._call_model(messages)
        return json.loads(response.strip())

    def see(self, input, prompt: str) -> str:
        """Analyze an image with AI vision."""
        if isinstance(input, str):
            with open(input, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(input, bytes):
            image_data = base64.b64encode(input).decode("utf-8")
        else:
            image_data = str(input)

        config = get_config()
        provider = config["ai"]["provider"]

        if provider == "anthropic":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }]

        return self._call_model(messages)

    def predict(self, prompt: str, schema=None) -> object:
        """Make a prediction with confidence score."""
        predict_prompt = (
            f"{prompt}\n\nProvide your prediction as JSON with "
            f'"value" (your prediction) and "confidence" (0.0 to 1.0).'
        )
        messages = [{"role": "user", "content": predict_prompt}]
        response = self._call_model(messages)

        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        data = json.loads(text)
        return ConfidentValue(value=data["value"], confidence=data["confidence"])

    def enrich(self, items: list, prompt: str) -> list:
        """Enrich a list of items using AI. Used in pipelines."""
        if not items:
            return []

        result = []
        for item in items:
            msg = f"{prompt}\n\nItem: {json.dumps(item)}"
            messages = [{"role": "user", "content": msg}]
            response = self._call_model(messages)

            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.rstrip().endswith("```"):
                    text = text.rstrip()[:-3]
                text = text.strip()

            try:
                enrichment = json.loads(text)
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.update(enrichment)
                    result.append(_to_drift_dict(merged))
                else:
                    result.append(item)
            except json.JSONDecodeError:
                if isinstance(item, dict):
                    merged = dict(item)
                    merged["enrichment"] = response
                    result.append(_to_drift_dict(merged))
                else:
                    result.append(item)

        return result

    def score(self, items: list, prompt: str) -> list:
        """Score a list of items using AI. Used in pipelines."""
        if not items:
            return []

        result = []
        for item in items:
            msg = f"{prompt}\n\nItem: {json.dumps(item)}\n\nRespond with only a number."
            messages = [{"role": "user", "content": msg}]
            response = self._call_model(messages)

            try:
                score_val = float(response.strip())
            except ValueError:
                score_val = 0

            if isinstance(item, dict):
                scored = dict(item)
                scored["score"] = score_val
                result.append(_to_drift_dict(scored))
            else:
                result.append(item)

        return result
