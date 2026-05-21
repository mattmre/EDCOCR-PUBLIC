"""Management command to display worker fleet status.

Shows current state of all registered workers, active jobs, and
queue summary information.

Usage:
    python manage.py fleet_status
"""

from django.core.management.base import BaseCommand
from django.db.models import Count
from django.utils import timezone

from jobs.models import Job, Worker


class Command(BaseCommand):
    help = "Display current worker fleet and job status"

    def handle(self, *args, **options):
        now = timezone.now()

        # Worker summary — single aggregated query
        workers = Worker.objects.all()
        worker_status_counts = dict(
            Worker.objects.values_list("status").annotate(c=Count("pk")).values_list("status", "c")
        )
        total_workers = sum(worker_status_counts.values())

        self.stdout.write("=" * 60)
        self.stdout.write("  Fleet Status")
        self.stdout.write("=" * 60)

        self.stdout.write("")
        self.stdout.write(f"  Workers: {total_workers} total")
        self.stdout.write(f"    Online:   {worker_status_counts.get(Worker.Status.ONLINE, 0)}")
        self.stdout.write(f"    Busy:     {worker_status_counts.get(Worker.Status.BUSY, 0)}")
        self.stdout.write(f"    Draining: {worker_status_counts.get(Worker.Status.DRAINING, 0)}")
        self.stdout.write(f"    Offline:  {worker_status_counts.get(Worker.Status.OFFLINE, 0)}")

        if workers.exists():
            self.stdout.write("")
            self.stdout.write(
                f"  {'Hostname':<30s} {'Status':<10s} {'GPU':<6s} {'Last Heartbeat':<20s}"
            )
            self.stdout.write(
                f"  {'-' * 30} {'-' * 10} {'-' * 6} {'-' * 20}"
            )
            for w in workers:
                if w.last_heartbeat:
                    age = int((now - w.last_heartbeat).total_seconds())
                    hb_age = f"{age}s ago"
                else:
                    hb_age = "never"
                gpu = "Yes" if w.gpu_available else "No"
                self.stdout.write(
                    f"  {w.hostname[:30]:<30s} {w.status:<10s} {gpu:<6s} {hb_age:<20s}"
                )

        # Job summary — single aggregated query
        job_status_counts = dict(
            Job.objects.values_list("status").annotate(c=Count("pk")).values_list("status", "c")
        )
        total_jobs = sum(job_status_counts.values())

        self.stdout.write("")
        self.stdout.write("  Jobs:")
        for status_choice in Job.Status:
            count = job_status_counts.get(status_choice.value, 0)
            if count > 0:
                self.stdout.write(f"    {status_choice.label + ':':<12s} {count}")

        self.stdout.write(f"    {'Total:':<12s} {total_jobs}")

        # Active jobs detail
        active_jobs = Job.objects.filter(
            status__in=[Job.Status.PROCESSING, Job.Status.ASSEMBLING, Job.Status.INGESTING]
        )[:5]
        if active_jobs.exists():
            self.stdout.write("")
            self.stdout.write("  Active Jobs:")
            for job in active_jobs:
                pct = 0
                if job.total_pages > 0:
                    pct = int(job.pages_completed / job.total_pages * 100)
                self.stdout.write(
                    f"    {str(job.job_id)[:8]}  {job.status}  "
                    f"{job.pages_completed}/{job.total_pages} pages ({pct}%)"
                )

        self.stdout.write("")
        self.stdout.write("=" * 60)
