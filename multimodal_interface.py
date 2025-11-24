#!/usr/bin/env python3
"""
Multimodal Interface for Vision LLMs
Supports GPT-4V, GPT-4o, and Claude 3.5 Sonnet with vision.
"""

import os
import base64
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class MultimodalMessage:
    """Container for multimodal message with text and images."""
    
    def __init__(self, text: str, images: List[str] = None):
        """
        Initialize multimodal message.
        
        Args:
            text: Text content
            images: List of base64-encoded images
        """
        self.text = text
        self.images = images or []
    
    def add_image(self, image_base64: str):
        """Add an image to the message."""
        self.images.append(image_base64)


class MultimodalInterface:
    """Interface for vision-enabled LLMs."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize multimodal interface.
        
        Args:
            model: Model name (gpt-4o, gpt-4-vision-preview, claude-3-5-sonnet-20241022)
            api_key: API key (if None, loads from environment)
        """
        self.model = model
        self.api_key = api_key
        
        # Determine provider
        if "gpt" in model.lower():
            self.provider = "openai"
            if api_key is None:
                self.api_key = os.getenv("OPENAI_API_KEY")
        elif "claude" in model.lower():
            self.provider = "anthropic"
            if api_key is None:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        if not self.api_key:
            raise ValueError(f"API key not found for {self.provider}")
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def send_message(self, message: MultimodalMessage, 
                    system_prompt: Optional[str] = None,
                    temperature: float = 0,
                    max_tokens: int = 4000) -> str:
        """
        Send multimodal message to LLM.
        
        Args:
            message: MultimodalMessage with text and images
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text from LLM
        """
        if self.provider == "openai":
            return self._send_openai(message, system_prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._send_anthropic(message, system_prompt, temperature, max_tokens)
    
    def _send_openai(self, message: MultimodalMessage, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
        """Send message to OpenAI API."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build user message with text and images
        content = []
        
        # Add text
        content.append({"type": "text", "text": message.text})
        
        # Add images
        for img_base64 in message.images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "high"  # Use high detail for geometric precision
                }
            })
        
        messages.append({"role": "user", "content": content})
        
        # Call API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _send_anthropic(self, message: MultimodalMessage, system_prompt: Optional[str],
                       temperature: float, max_tokens: int) -> str:
        """Send message to Anthropic API."""
        content = []
        
        # Add images first (Anthropic prefers images before text)
        for img_base64 in message.images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            })
        
        # Add text
        content.append({"type": "text", "text": message.text})
        
        # Call API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        return response.content[0].text
    
    def send_conversation(self, messages: List[Dict[str, Any]], 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0,
                         max_tokens: int = 4000) -> str:
        """
        Send multi-turn conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Response text
        """
        if self.provider == "openai":
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=messages
            )
            return response.content[0].text


# Test function
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("Multimodal Interface Test")
    print("="*70)
    print()
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"OpenAI API Key: {'✓ Found' if openai_key else '✗ Not found'}")
    print(f"Anthropic API Key: {'✓ Found' if anthropic_key else '✗ Not found'}")
    print()
    
    if not openai_key and not anthropic_key:
        print("No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        sys.exit(1)
    
    # Test with available provider
    if openai_key:
        print("Testing OpenAI GPT-4o...")
        interface = MultimodalInterface(model="gpt-4o")
        
        message = MultimodalMessage(text="Hello! Can you describe geometric shapes?")
        response = interface.send_message(message)
        
        print(f"Response: {response[:200]}...")
        print("✓ OpenAI test successful")
    
    elif anthropic_key:
        print("Testing Anthropic Claude 3.5 Sonnet...")
        interface = MultimodalInterface(model="claude-3-5-sonnet-20241022")
        
        message = MultimodalMessage(text="Hello! Can you describe geometric shapes?")
        response = interface.send_message(message)
        
        print(f"Response: {response[:200]}...")
        print("✓ Anthropic test successful")

