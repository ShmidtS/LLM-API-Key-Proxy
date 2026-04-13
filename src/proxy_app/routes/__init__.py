# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from proxy_app.routes.chat import router as chat_router
from proxy_app.routes.anthropic import router as anthropic_router
from proxy_app.routes.embeddings import router as embeddings_router
from proxy_app.routes.models import router as models_router
from proxy_app.routes.admin import router as admin_router
from proxy_app.routes.images import router as images_router
from proxy_app.routes.audio import router as audio_router
from proxy_app.routes.responses import router as responses_router
from proxy_app.routes.tools import router as tools_router
from proxy_app.routes.moderation import router as moderation_router
from proxy_app.routes.batches import router as batches_router
from proxy_app.routes.files import router as files_router

all_routers = [
    chat_router,
    anthropic_router,
    embeddings_router,
    models_router,
    admin_router,
    images_router,
    audio_router,
    responses_router,
    tools_router,
    moderation_router,
    batches_router,
    files_router,
]
