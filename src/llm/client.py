import os
import re
import json
import logging
from typing import Optional
from google import genai
from google.genai import types
from openai import OpenAI

logger = logging.getLogger(__name__)

GENAI_AVAILABLE = True
try:
    import google.genai
except ImportError:
    GENAI_AVAILABLE = False


def sanitize_json(raw_text: str) -> str:
    """Strips markdown code blocks from the raw LLM output before JSON parsing."""
    text = raw_text.strip()
    
    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx + 1:]
        
        if text.endswith("```"):
            text = text[:-3]
            
    return text.strip()


class LLMClient:
    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")
        
        self.gemini_client = None
        if self.provider == "gemini" and GENAI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                try:
                    self.gemini_client = genai.Client(api_key=api_key)
                except Exception as e:
                    logger.warning("Could not initialize genai.Client: %s", e)
                    
        self.openai_client = None
        if self.provider == "ollama":
            try:
                self.openai_client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama", # required but ignored
                )
            except Exception as e:
                logger.warning("Could not initialize OpenAI client for Ollama: %s", e)
                
    @property
    def is_configured(self) -> bool:
        if self.provider == "ollama":
            return self.openai_client is not None
        return self.gemini_client is not None

    def generate_text(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model: str = "gemini-3.1-flash-preview", 
        response_format: str = "text",
        temperature: float = 0.1
    ) -> str:
        """Routes text generation request to the selected LLM provider."""
        
        if self.provider == "ollama" and self.openai_client:
            return self._generate_ollama(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.ollama_model,
                response_format=response_format,
                temperature=temperature
            )
            
        if self.gemini_client:
            return self._generate_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                response_format=response_format,
                temperature=temperature
            )
            
        raise ValueError(f"No configured client available for provider: {self.provider}")

    def _generate_gemini(self, system_prompt: str, user_prompt: str, model: str, response_format: str, temperature: float) -> str:
        mime_type = "application/json" if response_format == "json" else "text/plain"
        
        response = self.gemini_client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type=mime_type,
                temperature=temperature,
            ),
        )
        raw_text = response.text or ""
        
        if response_format == "json":
            return sanitize_json(raw_text)
        return raw_text

    def _generate_ollama(self, system_prompt: str, user_prompt: str, model: str, response_format: str, temperature: float) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": {"options": {"num_ctx": 8192}}
        }
        
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
            
        response = self.openai_client.chat.completions.create(**kwargs)
        
        raw_text = response.choices[0].message.content or ""
        
        if response_format == "json":
            return sanitize_json(raw_text)
        return raw_text
