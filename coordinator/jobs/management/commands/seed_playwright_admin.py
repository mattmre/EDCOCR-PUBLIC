import json

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from jobs.models import CustodyEvent, Job, PageResult, Worker

SEED_PREFIX = "playwright-"
SEED_FILES = {
    "failed_job": "/seed/playwright-failed.pdf",
    "processing_job": "/seed/playwright-processing.pdf",
    "completed_job": "/seed/playwright-completed.pdf",
}
SEED_WORKERS = {
    "online_worker": "playwright-worker-online",
    "busy_worker": "playwright-worker-busy",
}
SEED_DOCUMENT_ID = "playwright-doc-001"


class Command(BaseCommand):
    help = "Seed deterministic coordinator admin data for Playwright workflows."

    def add_arguments(self, parser):
        parser.add_argument("--username", default="playwright-admin")
        parser.add_argument("--password")
        parser.add_argument("--email", default="playwright-admin@example.com")
        parser.add_argument("--reset", action="store_true")
        parser.add_argument("--json", action="store_true")

    def handle(self, *args, **options):
        self._validate_environment(options)

        if options["reset"]:
            self._reset_seeded_records()

        user = self._ensure_superuser(
            username=options["username"],
            password=options["password"],
            email=options["email"],
        )

        jobs = self._ensure_jobs()
        workers = self._ensure_workers()

        payload = {
            "username": user.username,
            "jobs": jobs,
            "workers": workers,
            "seeded": True,
        }

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
            return

        self.stdout.write("Seeded Playwright admin data:")
        self.stdout.write(f"  superuser: {user.username}")
        for label, source_file in SEED_FILES.items():
            self.stdout.write(f"  {label}: {source_file}")
        for label, hostname in SEED_WORKERS.items():
            self.stdout.write(f"  {label}: {hostname}")

    def _validate_environment(self, options):
        if not options["password"]:
            raise CommandError(
                "seed_playwright_admin requires an explicit --password value."
            )

        if getattr(settings, "DEPLOYMENT_ENV", "").strip().lower() == "production":
            raise CommandError(
                "seed_playwright_admin cannot run when DEPLOYMENT_ENV=production."
            )

    def _reset_seeded_records(self):
        Job.objects.filter(source_file__startswith="/seed/playwright-").delete()
        Worker.objects.filter(hostname__startswith=SEED_PREFIX).delete()

    def _ensure_superuser(self, username, password, email):
        user_model = get_user_model()
        user, _ = user_model.objects.get_or_create(
            username=username,
            defaults={
                "email": email,
                "is_staff": True,
                "is_superuser": True,
            },
        )
        if not user.is_staff:
            user.is_staff = True
        if not user.is_superuser:
            user.is_superuser = True
        if email and user.email != email:
            user.email = email
        user.set_password(password)
        user.save()
        return user

    def _ensure_jobs(self):
        failed_job, _ = Job.objects.update_or_create(
            source_file=SEED_FILES["failed_job"],
            defaults={
                "status": Job.Status.FAILED,
                "priority": Job.Priority.NORMAL,
                "total_pages": 8,
                "pages_completed": 2,
                "pages_failed": 1,
                "detected_language": "en",
                "error_message": "Playwright seeded failure",
                "assigned_worker": SEED_WORKERS["busy_worker"],
            },
        )
        processing_job, _ = Job.objects.update_or_create(
            source_file=SEED_FILES["processing_job"],
            defaults={
                "status": Job.Status.PROCESSING,
                "priority": Job.Priority.URGENT,
                "total_pages": 12,
                "pages_completed": 5,
                "pages_failed": 0,
                "detected_language": "en",
                "assigned_worker": SEED_WORKERS["busy_worker"],
            },
        )
        completed_job, _ = Job.objects.update_or_create(
            source_file=SEED_FILES["completed_job"],
            defaults={
                "status": Job.Status.COMPLETED,
                "priority": Job.Priority.NORMAL,
                "total_pages": 4,
                "pages_completed": 4,
                "pages_failed": 0,
                "detected_language": "en",
                "result_summary": {"seeded": True},
                "assigned_worker": SEED_WORKERS["online_worker"],
            },
        )

        for page_num in (1, 2):
            PageResult.objects.update_or_create(
                job=completed_job,
                page_num=page_num,
                defaults={
                    "document_id": SEED_DOCUMENT_ID,
                    "ocr_method": "paddle",
                    "ocr_language": "en",
                    "ocr_confidence": 0.98,
                    "text_length": 120 + page_num,
                    "status": "completed",
                    "worker_hostname": SEED_WORKERS["online_worker"],
                    "processing_time_ms": 220 + page_num,
                },
            )

        CustodyEvent.objects.update_or_create(
            job=completed_job,
            document_id=SEED_DOCUMENT_ID,
            event_type="ingested",
            defaults={
                "worker_hostname": SEED_WORKERS["online_worker"],
                "data": {"seeded": True, "phase": "ingest"},
                "prev_hash": "",
                "event_hash": "a" * 64,
                "chain_finalized": False,
            },
        )
        CustodyEvent.objects.update_or_create(
            job=completed_job,
            document_id=SEED_DOCUMENT_ID,
            event_type="completed",
            defaults={
                "worker_hostname": SEED_WORKERS["online_worker"],
                "data": {"seeded": True, "phase": "complete"},
                "prev_hash": "a" * 64,
                "event_hash": "b" * 64,
                "chain_finalized": True,
            },
        )

        return {
            "failed_job_id": str(failed_job.job_id),
            "processing_job_id": str(processing_job.job_id),
            "completed_job_id": str(completed_job.job_id),
        }

    def _ensure_workers(self):
        online_worker, _ = Worker.objects.update_or_create(
            hostname=SEED_WORKERS["online_worker"],
            defaults={
                "status": Worker.Status.ONLINE,
                "queues": ["ocr_gpu"],
                "capabilities": ["ocr", "classify"],
                "gpu_available": True,
                "gpu_model": "Seed GPU",
                "gpu_vram_mb": 8192,
                "tasks_completed": 11,
                "tasks_failed": 1,
            },
        )
        busy_worker, _ = Worker.objects.update_or_create(
            hostname=SEED_WORKERS["busy_worker"],
            defaults={
                "status": Worker.Status.BUSY,
                "queues": ["ocr_gpu"],
                "capabilities": ["ocr"],
                "gpu_available": False,
                "tasks_completed": 7,
                "tasks_failed": 2,
                "current_task_id": "playwright-seeded-task",
            },
        )
        return {
            "online_worker": online_worker.hostname,
            "busy_worker": busy_worker.hostname,
        }
