from typing import Dict, List, Optional
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class OpenRouterAction:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.3-70b-instruct:free"

    def generate(self, prompt: str) -> str:
        """Generate a response using OpenRouter API"""
        if not self.api_key:
            return "Error: OpenRouter API key not configured"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://medical-document-assistant.example.com",
            "X-Title": "Medical Document Assistant"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a medical document assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                return "Error: Unexpected API response format"
            else:
                return f"Error: API returned status code {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"

def get_actions() -> Dict:
    """Return available actions for guardrails"""
    return {
        "openrouter_generate": OpenRouterAction().generate
    } 