"use client";

import { useCallback } from "react";
import { Tabs } from "@/components/Tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { AlertsList } from "@/components/AlertsList";
import { AlertRulesList } from "@/components/AlertRulesList";
import { NotificationChannelsList } from "@/components/NotificationChannelsList";
import { useRequireAuth } from "@/lib/auth";
import {
  useActiveAlerts,
  useAlertRules,
  useNotificationChannels,
} from "@/lib/hooks";
import { ApiError } from "@/lib/api-client";
import {
  muteAlert,
  unmuteAlert,
  testChannel,
} from "@/lib/alerts-api";

type ApiOutcome =
  | { kind: "ok" }
  | { kind: "forbidden" }
  | { kind: "not_provisioned" }
  | { kind: "error"; message: string };

function classifyError(err: Error | null): ApiOutcome {
  if (!err) return { kind: "ok" };
  const status = (err as { status?: number }).status;
  if (status === 403) return { kind: "forbidden" };
  if (status === 404 || status === 501) return { kind: "not_provisioned" };
  return { kind: "error", message: err.message };
}

interface FallbackProps {
  outcome: ApiOutcome;
}

function ApiOutcomeFallback({ outcome }: FallbackProps) {
  if (outcome.kind === "ok") return null;
  if (outcome.kind === "forbidden") {
    return (
      <Card data-testid="alerts-forbidden">
        <CardHeader>
          <CardTitle>Platform admin role required</CardTitle>
          <CardDescription>
            Your API key does not have permission to view alerts.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Ask a platform admin to grant the <code>alerts:read</code> scope.
          </p>
        </CardContent>
      </Card>
    );
  }
  if (outcome.kind === "not_provisioned") {
    return (
      <Card data-testid="alerts-not-provisioned">
        <CardHeader>
          <CardTitle>Alerts API not yet provisioned</CardTitle>
          <CardDescription>
            The admin alerts endpoint is not available on this deployment. Use
            Prometheus / Grafana directly while this is being rolled out.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm">
            <a
              href="/grafana"
              className="text-primary hover:underline"
              data-testid="alerts-grafana-link"
            >
              Open Grafana →
            </a>
          </p>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card data-testid="alerts-error">
      <CardHeader>
        <CardTitle>Failed to load alerts</CardTitle>
        <CardDescription>{outcome.message}</CardDescription>
      </CardHeader>
    </Card>
  );
}

export default function AlertsPage() {
  useRequireAuth();

  const alerts = useActiveAlerts(true);
  const rules = useAlertRules();
  const channels = useNotificationChannels();

  const alertsOutcome = classifyError(alerts.error);
  const rulesOutcome = classifyError(rules.error);
  const channelsOutcome = classifyError(channels.error);

  const handleMute = useCallback(
    async (alertId: string, payload: { reason: string }) => {
      try {
        await muteAlert(alertId, payload);
      } catch (err) {
        if (err instanceof ApiError) {
          throw err;
        }
        throw err;
      } finally {
        alerts.refresh();
      }
    },
    [alerts]
  );

  const handleUnmute = useCallback(
    async (alertId: string) => {
      try {
        await unmuteAlert(alertId);
      } finally {
        alerts.refresh();
      }
    },
    [alerts]
  );

  const handleTest = useCallback(async (channelId: string) => {
    const result = await testChannel(channelId);
    return { ok: result.ok, message: result.message };
  }, []);

  const tabs = [
    {
      id: "active",
      label: "Active alerts",
      content: (
        <Card>
          <CardHeader>
            <CardTitle>Currently firing</CardTitle>
            <CardDescription>
              Auto-refreshes every 15 seconds. Mute action only suppresses
              operator notifications -- the underlying breach keeps being
              recorded in the audit log.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {alertsOutcome.kind !== "ok" ? (
              <ApiOutcomeFallback outcome={alertsOutcome} />
            ) : (
              <AlertsList
                alerts={alerts.data ?? []}
                loading={alerts.loading}
                onMute={handleMute}
                onUnmute={handleUnmute}
              />
            )}
          </CardContent>
        </Card>
      ),
    },
    {
      id: "rules",
      label: "Alert rules",
      content: (
        <Card>
          <CardHeader>
            <CardTitle>Configured rules</CardTitle>
            <CardDescription>
              Operator-tunable thresholds for the queue depth and per-tenant
              SLA monitors. PromQL is read-only.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {rulesOutcome.kind !== "ok" ? (
              <ApiOutcomeFallback outcome={rulesOutcome} />
            ) : (
              <AlertRulesList rules={rules.data ?? []} loading={rules.loading} />
            )}
          </CardContent>
        </Card>
      ),
    },
    {
      id: "channels",
      label: "Notification channels",
      content: (
        <Card>
          <CardHeader>
            <CardTitle>Notification channels</CardTitle>
            <CardDescription>
              Webhook, Slack, and email destinations. "Test" sends a synthetic
              alert through the channel; results are advisory.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {channelsOutcome.kind !== "ok" ? (
              <ApiOutcomeFallback outcome={channelsOutcome} />
            ) : (
              <NotificationChannelsList
                channels={channels.data ?? []}
                loading={channels.loading}
                onTest={handleTest}
              />
            )}
          </CardContent>
        </Card>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Alerts</h1>
        <p className="text-sm text-muted-foreground">
          Active alerts, rule thresholds, and notification channels. Backed by{" "}
          <code>api/queue_alerting.py</code> and <code>sla_monitoring.py</code>.
        </p>
      </div>
      <Tabs tabs={tabs} defaultTab="active" />
    </div>
  );
}
