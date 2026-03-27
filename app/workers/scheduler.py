"""APScheduler setup.

Creates and configures the AsyncIOScheduler used by the application.
Jobs are registered separately in ``app.workers.jobs``.

Exports:
    create_scheduler -- factory function
"""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore


def create_scheduler() -> AsyncIOScheduler:
    """Create and return a configured AsyncIOScheduler instance.

    Uses an in-memory job store (no persistence across restarts).
    Timezone is set to UTC.

    Returns:
        Configured but not yet started AsyncIOScheduler.
    """
    jobstores = {
        "default": MemoryJobStore(),
    }
    job_defaults = {
        "coalesce": True,       # Run only once if multiple executions were missed
        "max_instances": 1,     # Prevent overlapping job executions
    }

    scheduler = AsyncIOScheduler(
        jobstores=jobstores,
        job_defaults=job_defaults,
        timezone="UTC",
    )

    return scheduler


# Module-level singleton — imported by dashboard.py and main.py
scheduler: AsyncIOScheduler = create_scheduler()
