"""Auto-load GLNIS runtime shims when Python imports sitecustomize."""

from __future__ import annotations

import glnis_runtime_bootstrap

glnis_runtime_bootstrap.bootstrap(allow_reexec=False)
