"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { Tabs } from "@/components/Tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TenantConfigForm } from "@/components/TenantConfigForm";
import { GlossaryEditor } from "@/components/GlossaryEditor";
import { useRequireAuth } from "@/lib/auth";
import { useTenantConfig } from "@/lib/hooks";
import { upsertTenantConfig } from "@/lib/tenant-api";
import { ApiError } from "@/lib/api-client";
import type { TenantConfigUpdate } from "@/lib/types";

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

export default function TenantDetailPage() {
  useRequireAuth();
  const params = useParams<{ tenantId: string }>();
  const tenantId = decodeURIComponent((params?.tenantId ?? "") as string);
  const config = useTenantConfig(tenantId);
  const [saving, setSaving] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  async function handleSave(payload: TenantConfigUpdate) {
    setSaving(true);
    setSubmitError(null);
    try {
      await upsertTenantConfig(tenantId, payload);
      config.refresh();
    } catch (err) {
      setSubmitError(describeError(err));
    } finally {
      setSaving(false);
    }
  }

  const tabs = [
    {
      id: "config",
      label: "Config",
      content: (
        <Card>
          <CardHeader>
            <CardTitle>Translation policy</CardTitle>
            <CardDescription>
              Routes through{" "}
              <code className="rounded bg-muted px-1">/api/v1/translation/tenants/{tenantId}/config</code>.
              {" "}Saved values take effect on the next translation request.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {config.loading && !config.data ? (
              <p className="text-sm text-muted-foreground">Loading…</p>
            ) : config.error ? (
              <p className="text-sm text-destructive" role="alert">
                {describeError(config.error)}
              </p>
            ) : (
              <TenantConfigForm
                tenantId={tenantId}
                initial={config.data}
                onSubmit={handleSave}
                submitError={submitError}
                saving={saving}
              />
            )}
          </CardContent>
        </Card>
      ),
    },
    {
      id: "glossary",
      label: "Glossary",
      content: (
        <Card>
          <CardHeader>
            <CardTitle>Tenant glossary</CardTitle>
            <CardDescription>
              CRUD against{" "}
              <code className="rounded bg-muted px-1">
                /api/v1/translation/tenants/{tenantId}/glossary
              </code>
              . Inline edits use optimistic UI; failures roll back automatically.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <GlossaryEditor tenantId={tenantId} />
          </CardContent>
        </Card>
      ),
    },
    {
      id: "custody",
      label: "Custody",
      content: (
        <Card>
          <CardHeader>
            <CardTitle>Tenant custody events</CardTitle>
            <CardDescription>
              Translation custody events filtered by tenant. Coming soon — for now
              please use the per-job audit timeline.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground">
              The translation router emits custody events scoped to this tenant
              (e.g. <code className="rounded bg-muted px-1">TRANSLATION_REJECTED</code>
              {" "}on policy denial). A tenant-scoped custody listing endpoint is not
              yet exposed; until it lands you can locate the events on a per-job
              audit page.
            </p>
            <p className="text-sm">
              <Link href="/audit" className="text-primary hover:underline">
                Open the audit timeline →
              </Link>
            </p>
          </CardContent>
        </Card>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <Link
          href="/admin/tenants"
          className="text-xs text-muted-foreground hover:underline"
        >
          ← Tenants
        </Link>
        <h1 className="mt-1 text-2xl font-semibold">
          <span className="font-mono" data-testid="tenant-detail-id">
            {tenantId}
          </span>
        </h1>
        <p className="text-sm text-muted-foreground">
          Translation config, glossary, and tenant-scoped custody events.
        </p>
      </div>
      <Tabs tabs={tabs} defaultTab="config" />
    </div>
  );
}
