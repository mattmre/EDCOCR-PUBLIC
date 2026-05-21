"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { AlertRuleEditor } from "@/components/AlertRuleEditor";
import { useRequireAuth } from "@/lib/auth";
import { useAlertRule, useNotificationChannels } from "@/lib/hooks";
import { ApiError } from "@/lib/api-client";
import { updateRuleThreshold } from "@/lib/alerts-api";
import type { AlertRuleUpdate } from "@/lib/types";

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

export default function AlertRuleDetailPage() {
  useRequireAuth();
  const params = useParams<{ ruleId: string }>();
  const ruleId = decodeURIComponent((params?.ruleId ?? "") as string);

  const rule = useAlertRule(ruleId);
  const channels = useNotificationChannels();

  const [saving, setSaving] = useState<boolean>(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const handleSubmit = useCallback(
    async (payload: AlertRuleUpdate) => {
      setSaving(true);
      setSaveError(null);
      try {
        await updateRuleThreshold(ruleId, payload);
        rule.refresh();
      } catch (err) {
        setSaveError(describeError(err));
      } finally {
        setSaving(false);
      }
    },
    [ruleId, rule]
  );

  const errStatus = (rule.error as { status?: number } | null)?.status;

  return (
    <div className="space-y-6">
      <div>
        <Link
          href="/admin/alerts"
          className="text-xs text-muted-foreground hover:underline"
        >
          ← Alerts
        </Link>
        <h1 className="mt-1 text-2xl font-semibold">
          <span className="font-mono" data-testid="alert-rule-id">
            {ruleId}
          </span>
        </h1>
        <p className="text-sm text-muted-foreground">
          Edit operator-tunable thresholds. PromQL is read-only.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{rule.data?.name ?? ruleId}</CardTitle>
          <CardDescription>
            {rule.data?.description ??
              "Tune the threshold, evaluation window, severity, and notification routing."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {rule.loading && !rule.data ? (
            <p className="text-sm text-muted-foreground">Loading rule…</p>
          ) : rule.error ? (
            errStatus === 403 ? (
              <p
                className="text-sm text-destructive"
                role="alert"
                data-testid="rule-forbidden"
              >
                Platform admin role required.
              </p>
            ) : errStatus === 404 || errStatus === 501 ? (
              <p
                className="text-sm text-muted-foreground"
                role="alert"
                data-testid="rule-not-provisioned"
              >
                Alert rule not found or alerts API not provisioned.
              </p>
            ) : (
              <p
                className="text-sm text-destructive"
                role="alert"
                data-testid="rule-error"
              >
                {describeError(rule.error)}
              </p>
            )
          ) : rule.data ? (
            <AlertRuleEditor
              rule={rule.data}
              channels={channels.data ?? []}
              saving={saving}
              saveError={saveError}
              onSubmit={handleSubmit}
            />
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
