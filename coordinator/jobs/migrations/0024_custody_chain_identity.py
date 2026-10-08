"""Persist immutable custody-chain identities before Job retention can unlink rows."""

import hashlib

from django.db import migrations, models


def _chain_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def backfill_chain_identity(apps, schema_editor):
    # CustodyEvent has FORCE RLS before this migration. The intended
    # NOBYPASS migration role must opt into the existing control-plane policy
    # before it can observe or update any legacy chain rows.
    if schema_editor.connection.vendor == "postgresql":
        with schema_editor.connection.cursor() as cursor:
            cursor.execute("SET LOCAL app.control_plane = 'on'")

    CustodyEvent = apps.get_model("jobs", "CustodyEvent")
    CustodyArchiveHead = apps.get_model("jobs", "CustodyArchiveHead")
    hash_to_chain = {}
    pending = []
    for event in CustodyEvent.objects.order_by("id").iterator(chunk_size=2_000):
        previous = str(event.prev_hash or "").strip()
        event_hash = str(event.event_hash or "").strip()
        if previous and previous in hash_to_chain:
            chain_id = hash_to_chain[previous]
        elif event.job_id:
            chain_id = _chain_digest(f"job\0{event.job_id}\0{event.document_id}")
        elif previous:
            # The predecessor may already be in an earlier archive. Its hash
            # is still an immutable, collision-resistant continuation anchor.
            chain_id = _chain_digest(f"legacy-anchor\0{previous}")
        elif event_hash:
            # Distinct deleted jobs can share document_id. A sealed genesis
            # hash keeps those otherwise-unrecoverable legacy chains separate.
            chain_id = _chain_digest(f"legacy-genesis\0{event_hash}")
        else:
            chain_id = _chain_digest(f"legacy-row\0{event.id}")
        event.chain_id = chain_id
        pending.append(event)
        if event_hash:
            hash_to_chain[event_hash] = chain_id
        if len(pending) >= 2_000:
            CustodyEvent.objects.bulk_update(pending, ["chain_id"])
            pending.clear()
    if pending:
        CustodyEvent.objects.bulk_update(pending, ["chain_id"])

    # If a retained chain begins with a nonblank predecessor, older rows were
    # already removed before archive-head manifests existed. Preserve that tip
    # explicitly, but label the historical provenance gap as unverified.
    seen = set()
    heads = []
    for event in CustodyEvent.objects.order_by("id").iterator(chunk_size=2_000):
        if event.chain_id in seen:
            continue
        seen.add(event.chain_id)
        previous = str(event.prev_hash or "").strip()
        if previous:
            heads.append(
                CustodyArchiveHead(
                    chain_id=event.chain_id,
                    event_hash=previous,
                    last_event_id=0,
                    archive_path="legacy-pre-0024-unknown",
                    archive_checksum="",
                    legacy_unverified=True,
                )
            )
    if heads:
        CustodyArchiveHead.objects.bulk_create(heads, ignore_conflicts=True)

    if CustodyEvent.objects.filter(chain_id="").exists():
        raise RuntimeError("Custody chain identity backfill left blank chain_id rows")


def apply_archive_head_rls(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            ALTER TABLE jobs_custodyarchivehead ENABLE ROW LEVEL SECURITY;
            ALTER TABLE jobs_custodyarchivehead FORCE ROW LEVEL SECURITY;
            DROP POLICY IF EXISTS jobs_custodyarchivehead_control_plane
              ON jobs_custodyarchivehead;
            CREATE POLICY jobs_custodyarchivehead_control_plane
              ON jobs_custodyarchivehead
              USING (current_setting('app.control_plane', true) = 'on')
              WITH CHECK (current_setting('app.control_plane', true) = 'on');
            """
        )


class Migration(migrations.Migration):
    dependencies = [("jobs", "0023_legacy_child_control_plane_rls")]

    operations = [
        migrations.AddField(
            model_name="custodyevent",
            name="chain_id",
            field=models.CharField(
                blank=True,
                db_index=True,
                default="",
                help_text=(
                    "Immutable custody-chain identity independent of the Job "
                    "foreign key."
                ),
                max_length=64,
            ),
        ),
        migrations.AddField(
            model_name="custodyevent",
            name="hash_version",
            field=models.PositiveSmallIntegerField(
                default=1,
                help_text="Custody envelope version (1=legacy, 2=chain-bound).",
            ),
        ),
        migrations.CreateModel(
            name="CustodyArchiveHead",
            fields=[
                (
                    "chain_id",
                    models.CharField(max_length=64, primary_key=True, serialize=False),
                ),
                ("event_hash", models.CharField(max_length=64)),
                ("last_event_id", models.PositiveBigIntegerField(default=0)),
                (
                    "archive_path",
                    models.CharField(blank=True, default="", max_length=2048),
                ),
                (
                    "archive_checksum",
                    models.CharField(blank=True, default="", max_length=64),
                ),
                ("legacy_unverified", models.BooleanField(default=False)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.RunPython(backfill_chain_identity, migrations.RunPython.noop),
        migrations.RunPython(apply_archive_head_rls, migrations.RunPython.noop),
    ]
