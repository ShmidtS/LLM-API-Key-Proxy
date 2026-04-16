# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/provider_ui_config.py
"""
LiteLLM provider UI configuration for the rotator library.

This module handles:
- Known LiteLLM provider definitions (from scraped data) for UI display
- UI configuration (categories, notes, extra vars) for credential tool
"""

from typing import Dict, Any

# =============================================================================
# LiteLLM Provider UI Configuration
#
# Keys are route-based provider identifiers (e.g., "openai", "anthropic").
# Provider data (display_name, api_key_env_vars, etc.) comes from SCRAPED_PROVIDERS.
#
# This dict only contains UI-specific configuration:
#   - category: Provider category for display grouping
#   - note: (optional) Configuration notes shown to user
#   - extra_vars: (optional) Additional env vars to prompt for [(name, label, default), ...]
# =============================================================================

LITELLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # POPULAR - Most commonly used providers
    # =========================================================================
    "openai": {
        "category": "popular",
    },
    "anthropic": {
        "category": "popular",
    },
    "gemini": {
        "category": "popular",
    },
    "xai": {
        "category": "popular",
    },
    "deepseek": {
        "category": "popular",
    },
    "mistral": {
        "category": "popular",
    },
    "codestral": {
        "category": "popular",
    },
    "openrouter": {
        "category": "popular",
        "extra_vars": [
            ("OPENROUTER_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "kilocode": {
        "category": "popular",
        "extra_vars": [
            ("KILOCODE_API_BASE", "API Base URL", "https://kilo.ai/api/openrouter"),
        ],
    },
    "groq": {
        "category": "popular",
    },
    "chutes": {
        "category": "popular",
    },
    "nvidia": {
        "category": "popular",
        "extra_vars": [
            ("NVIDIA_API_BASE", "NIM API Base (optional)", None),
        ],
    },
    "perplexity": {
        "category": "popular",
    },
    "moonshot": {
        "category": "popular",
        "extra_vars": [
            ("MOONSHOT_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "zai": {
        "category": "popular",
        "extra_vars": [
            (
                "ZAI_API_BASE",
                "API Base URL (default: https://api.z.ai/api/coding/paas/v4)",
                "https://api.z.ai/api/coding/paas/v4",
            ),
        ],
    },
    "opencode": {
        "category": "popular",
        "extra_vars": [
            ("OPENCODE_API_BASE", "API Base URL", "https://opencode.ai/zen/v1"),
        ],
    },
    "trybons": {
        "category": "popular",
        "extra_vars": [
            ("TRYBONS_API_BASE", "API Base URL", "https://go.trybons.ai"),
        ],
    },
    "minimax": {
        "category": "popular",
        "extra_vars": [
            ("MINIMAX_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "xiaomi_mimo": {
        "category": "popular",
    },
    "nano-gpt": {
        "category": "popular",
    },
    "synthetic": {
        "category": "popular",
    },
    "colin": {
        "category": "popular",
        "note": "OpenAI Responses API format. Models: gpt-5.3-codex, gpt-4o.",
        "extra_vars": [
            ("COLIN_API_BASE", "API Base URL", "https://claude.colin1112.tech/v1"),
        ],
    },
    # =========================================================================
    # CLOUD PLATFORMS - Aggregators & cloud inference platforms
    # =========================================================================
    "together_ai": {
        "category": "cloud",
    },
    "fireworks_ai": {
        "category": "cloud",
        "extra_vars": [
            ("FIREWORKS_AI_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "replicate": {
        "category": "cloud",
    },
    "deepinfra": {
        "category": "cloud",
    },
    "anyscale": {
        "category": "cloud",
    },
    "baseten": {
        "category": "cloud",
    },
    "predibase": {
        "category": "cloud",
    },
    "novita": {
        "category": "cloud",
    },
    "featherless_ai": {
        "category": "cloud",
    },
    "hyperbolic": {
        "category": "cloud",
    },
    "lambda_ai": {
        "category": "cloud",
        "extra_vars": [
            ("LAMBDA_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "nebius": {
        "category": "cloud",
    },
    "galadriel": {
        "category": "cloud",
    },
    "friendliai": {
        "category": "cloud",
    },
    "sambanova": {
        "category": "cloud",
    },
    "cerebras": {
        "category": "cloud",
    },
    "meta_llama": {
        "category": "cloud",
    },
    "ai21": {
        "category": "cloud",
    },
    "cohere_chat": {
        "category": "cloud",
    },
    "alephalpha": {
        "category": "cloud",
    },
    "huggingface": {
        "category": "cloud",
    },
    "github": {
        "category": "cloud",
    },
    "helicone": {
        "category": "cloud",
        "note": "LLM gateway/proxy with analytics.",
    },
    "heroku": {
        "category": "cloud",
        "extra_vars": [
            (
                "HEROKU_API_BASE",
                "Heroku Inference URL",
                "https://us.inference.heroku.com",
            ),
        ],
    },
    "morph": {
        "category": "cloud",
    },
    "poe": {
        "category": "cloud",
    },
    "llamagate": {
        "category": "cloud",
    },
    "manus": {
        "category": "cloud",
    },
    # =========================================================================
    # ENTERPRISE / COMPLEX AUTH - Major cloud providers (may need extra config)
    # =========================================================================
    "azure": {
        "category": "enterprise",
        "note": "Requires Azure endpoint and API version.",
        "extra_vars": [
            ("AZURE_API_BASE", "Azure endpoint URL", None),
            ("AZURE_API_VERSION", "API version", "2024-02-15-preview"),
        ],
    },
    "azure_ai": {
        "category": "enterprise",
        "extra_vars": [
            ("AZURE_AI_API_BASE", "Azure AI endpoint URL", None),
        ],
    },
    "vertex_ai": {
        "category": "enterprise",
        "note": "Uses Google Cloud service account. Enter path to credentials JSON file.",
        "extra_vars": [
            ("VERTEXAI_PROJECT", "GCP Project ID", None),
            ("VERTEXAI_LOCATION", "GCP Location", "us-central1"),
        ],
    },
    "bedrock": {
        "category": "enterprise",
        "note": "Requires all three AWS credentials.",
        "extra_vars": [
            ("AWS_SECRET_ACCESS_KEY", "AWS Secret Access Key", None),
            ("AWS_REGION_NAME", "AWS Region", "us-east-1"),
        ],
    },
    "sagemaker": {
        "category": "enterprise",
        "note": "Requires all three AWS credentials.",
        "extra_vars": [
            ("AWS_SECRET_ACCESS_KEY", "AWS Secret Access Key", None),
            ("AWS_REGION_NAME", "AWS Region", "us-east-1"),
        ],
    },
    "databricks": {
        "category": "enterprise",
        "extra_vars": [
            ("DATABRICKS_API_BASE", "Databricks workspace URL", None),
        ],
    },
    "snowflake": {
        "category": "enterprise",
        "note": "Uses JWT authentication.",
        "extra_vars": [
            ("SNOWFLAKE_ACCOUNT_ID", "Snowflake Account ID", None),
        ],
    },
    "watsonx": {
        "category": "enterprise",
        "extra_vars": [
            ("WATSONX_URL", "watsonx.ai URL (optional)", None),
        ],
    },
    "cloudflare": {
        "category": "enterprise",
        "extra_vars": [
            ("CLOUDFLARE_ACCOUNT_ID", "Cloudflare Account ID", None),
        ],
    },
    "oci": {
        "category": "enterprise",
        "note": "Oracle Cloud Infrastructure. Requires OCI SDK configuration.",
    },
    "sap": {
        "category": "enterprise",
        "note": "SAP Generative AI Hub. Requires AI Core configuration.",
    },
    # =========================================================================
    # SPECIALIZED - Image, audio, embeddings, rerank providers
    # =========================================================================
    "stability": {
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "fal_ai": {
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "runwayml": {
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "recraft": {
        "category": "specialized",
        "note": "Image generation and editing.",
        "extra_vars": [
            ("RECRAFT_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "topaz": {
        "category": "specialized",
        "note": "Image enhancement provider.",
    },
    "elevenlabs": {
        "category": "specialized",
        "note": "Text-to-speech and audio transcription.",
    },
    "deepgram": {
        "category": "specialized",
        "note": "Audio transcription provider.",
    },
    "voyage": {
        "category": "specialized",
        "note": "Embeddings and rerank provider.",
    },
    "jina_ai": {
        "category": "specialized",
        "note": "Embeddings and rerank provider.",
    },
    "clarifai": {
        "category": "specialized",
    },
    "nlp_cloud": {
        "category": "specialized",
    },
    "milvus": {
        "category": "specialized",
        "note": "Vector database provider.",
        "extra_vars": [
            ("MILVUS_API_BASE", "Milvus Server URL", None),
        ],
    },
    "infinity": {
        "category": "specialized",
        "note": "Self-hosted embeddings/rerank server. API key is optional.",
        "extra_vars": [
            ("INFINITY_API_BASE", "Infinity Server URL", "http://localhost:8080"),
        ],
    },
    # =========================================================================
    # REGIONAL - Region-specific or specialized regional providers
    # =========================================================================
    "dashscope": {
        "category": "regional",
        "note": "Alibaba Cloud Qwen models.",
    },
    "volcengine": {
        "category": "regional",
        "note": "ByteDance cloud platform.",
    },
    "ovhcloud": {
        "category": "regional",
        "note": "European cloud provider.",
    },
    "nscale": {
        "category": "regional",
        "note": "EU sovereign cloud.",
    },
    # =========================================================================
    # LOCAL / SELF-HOSTED - Run locally or on your own infrastructure
    # =========================================================================
    "lm_studio": {
        "category": "local",
        "note": "Local provider. API key is optional. Start LM Studio server first.",
        "extra_vars": [
            ("LM_STUDIO_API_BASE", "API Base URL", "http://localhost:1234/v1"),
        ],
    },
    "hosted_vllm": {
        "category": "local",
        "note": "Self-hosted vLLM server. API key is optional.",
        "extra_vars": [
            ("HOSTED_VLLM_API_BASE", "vLLM Server URL", None),
        ],
    },
    "xinference": {
        "category": "local",
        "note": "Local Xinference server. API key is optional.",
        "extra_vars": [
            ("XINFERENCE_API_BASE", "Xinference URL", "http://127.0.0.1:9997/v1"),
        ],
    },
    "litellm_proxy": {
        "category": "local",
        "note": "Self-hosted LiteLLM Proxy gateway.",
        "extra_vars": [
            ("LITELLM_PROXY_API_BASE", "LiteLLM Proxy URL", "http://localhost:4000"),
        ],
    },
    "langgraph": {
        "category": "local",
        "note": "Self-hosted LangGraph server.",
        "extra_vars": [
            ("LANGGRAPH_API_BASE", "LangGraph URL", "http://localhost:2024"),
        ],
    },
    "ragflow": {
        "category": "local",
        "note": "Self-hosted RAGFlow server.",
        "extra_vars": [
            ("RAGFLOW_API_BASE", "RAGFlow URL", "http://localhost:9380"),
        ],
    },
    "docker_model_runner": {
        "category": "local",
        "note": "Local Docker Model Runner. API key is optional.",
        "extra_vars": [
            (
                "DOCKER_MODEL_RUNNER_API_BASE",
                "Docker Model Runner URL",
                "http://localhost:22088",
            ),
        ],
    },
    "lemonade": {
        "category": "local",
        "note": "Local proxy. API key is optional.",
        "extra_vars": [
            ("LEMONADE_API_BASE", "Lemonade URL", "http://localhost:8000/api/v1"),
        ],
    },
    # NOTE: ollama, llamafile, petals, triton are in PROVIDER_BLACKLIST
    # because they don't use standard API key authentication.
    # Use "Add Custom OpenAI-Compatible Provider" for these.
    # =========================================================================
    # OTHER - Miscellaneous providers
    # =========================================================================
    "aiml": {
        "category": "other",
        "extra_vars": [
            ("AIML_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "abliteration": {
        "category": "other",
    },
    "amazon_nova": {
        "category": "other",
    },
    "apertis": {
        "category": "other",
    },
    "bytez": {
        "category": "other",
    },
    "cometapi": {
        "category": "other",
    },
    "compactifai": {
        "category": "other",
    },
    "datarobot": {
        "category": "other",
        "extra_vars": [
            ("DATAROBOT_API_BASE", "DataRobot URL", "https://app.datarobot.com"),
        ],
    },
    "gradient_ai": {
        "category": "other",
        "extra_vars": [
            ("GRADIENT_AI_AGENT_ENDPOINT", "Gradient AI Endpoint (optional)", None),
        ],
    },
    "publicai": {
        "category": "other",
        "extra_vars": [
            ("PUBLICAI_API_BASE", "PublicAI URL", "https://platform.publicai.co/"),
        ],
    },
    "v0": {
        "category": "other",
    },
    "vercel_ai_gateway": {
        "category": "other",
    },
    "wandb": {
        "category": "other",
    },
}

# Category display order and labels
PROVIDER_CATEGORIES = [
    ("custom", "Custom (First-Party)"),
    ("custom_openai", "Custom OpenAI-Compatible"),
    ("popular", "Popular"),
    ("cloud", "Cloud Platforms"),
    ("enterprise", "Enterprise / Complex Auth"),
    ("specialized", "Specialized (Image/Audio/Embeddings)"),
    ("regional", "Regional"),
    ("local", "Local / Self-Hosted"),
    ("other", "Other"),
]
