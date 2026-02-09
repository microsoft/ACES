import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://secbench-azure-ai-services.cognitiveservices.azure.com/"
model_name = "gpt-5.2-codex"
deployment = "gpt-5.2-codex"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
api_version = "2025-04-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
)

response = client.responses.create(
    input=[
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_output_tokens=16384,
    model=deployment
)

print(response.output_text)
