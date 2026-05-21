"use client";

import { useState } from "react";
import { cn } from "@/lib/cn";
import { Button } from "@/components/ui/button";
import type { NotificationChannel, NotificationChannelType } from "@/lib/types";

/**
 * Best-effort partial redaction. The backend SHOULD return targets already
 * redacted, but this is a defense-in-depth client-side pass so an unredacted
 * value never leaks into the DOM if the API contract drifts.
 *
 *   slack:  preserves "#" prefix and last char of channel name
 *   email:  hides local part, keeps last char + "@example.com"
 *   webhook: keeps protocol + host, redacts path
 */
export function redactChannelTarget(
  type: NotificationChannelType,
  target: string
): string {
  if (!target) return "";
  if (type === "slack") {
    const trimmed = target.startsWith("#") ? target.slice(1) : target;
    if (trimmed.length === 0) return "#";
    if (trimmed.length <= 2) return `#${"*".repeat(Math.max(1, trimmed.length))}`;
    return `#${trimmed[0]}${"*".repeat(Math.max(3, trimmed.length - 2))}${
      trimmed[trimmed.length - 1]
    }`;
  }
  if (type === "email") {
    const at = target.indexOf("@");
    if (at < 0) return "***";
    const local = target.slice(0, at);
    const domain = target.slice(at);
    const localRedacted = local.length > 0 ? `***${local[local.length - 1]}` : "***";
    return `${localRedacted}${domain}`;
  }
  // webhook
  try {
    const url = new URL(target);
    return `${url.protocol}//${url.host}/***`;
  } catch {
    // Not a parseable URL -- redact most of it.
    if (target.length <= 6) return "***";
    return `${target.slice(0, 4)}***${target.slice(-2)}`;
  }
}

interface TestState {
  ok: boolean;
  at: number;
  message?: string;
}

export interface NotificationChannelsListProps {
  channels: NotificationChannel[];
  loading?: boolean;
  /** Called when the operator clicks "Test" on a channel row. */
  onTest?: (channelId: string) => Promise<{ ok: boolean; message?: string }>;
}

export function NotificationChannelsList({
  channels,
  loading,
  onTest,
}: NotificationChannelsListProps) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, TestState>>({});

  async function handleTest(id: string) {
    if (!onTest) return;
    setBusyId(id);
    try {
      const result = await onTest(id);
      setResults((r) => ({
        ...r,
        [id]: { ok: result.ok, at: Date.now(), message: result.message },
      }));
    } catch (err) {
      setResults((r) => ({
        ...r,
        [id]: {
          ok: false,
          at: Date.now(),
          message: err instanceof Error ? err.message : String(err),
        },
      }));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div
      className="overflow-x-auto rounded-md border border-border"
      data-testid="notification-channels-list"
    >
      <table className="w-full text-sm" aria-label="Notification channels">
        <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left">Type</th>
            <th className="px-3 py-2 text-left">Target</th>
            <th className="px-3 py-2 text-left">State</th>
            <th className="px-3 py-2 text-left">Last test</th>
            <th className="px-3 py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {loading && channels.length === 0 ? (
            <tr>
              <td
                colSpan={5}
                className="px-3 py-3 text-muted-foreground"
                data-testid="channels-loading"
              >
                Loading channels…
              </td>
            </tr>
          ) : channels.length === 0 ? (
            <tr>
              <td
                colSpan={5}
                className="px-3 py-3 text-muted-foreground"
                data-testid="channels-empty"
              >
                No notification channels configured.
              </td>
            </tr>
          ) : (
            channels.map((ch) => {
              const result = results[ch.id];
              const display = redactChannelTarget(ch.type, ch.target);
              return (
                <tr
                  key={ch.id}
                  className="border-t border-border hover:bg-muted/20"
                  data-testid={`channel-row-${ch.id}`}
                >
                  <td className="px-3 py-2 text-xs font-medium uppercase">{ch.type}</td>
                  <td
                    className="px-3 py-2 font-mono text-xs"
                    data-testid={`channel-target-${ch.id}`}
                  >
                    {display}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {ch.enabled ? (
                      <span className="text-green-700">enabled</span>
                    ) : (
                      <span className="text-muted-foreground">disabled</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {result ? (
                      <span
                        className={cn(
                          "inline-flex items-center gap-2",
                          result.ok ? "text-green-700" : "text-destructive"
                        )}
                        data-testid={`channel-test-result-${ch.id}`}
                      >
                        {result.ok ? "PASS" : "FAIL"}
                        <span className="text-muted-foreground">
                          {new Date(result.at).toLocaleTimeString()}
                        </span>
                        {result.message ? (
                          <span className="text-muted-foreground">— {result.message}</span>
                        ) : null}
                      </span>
                    ) : ch.last_test_at ? (
                      <span className="text-muted-foreground">
                        {ch.last_test_ok ? "PASS" : "FAIL"} ·{" "}
                        {new Date(ch.last_test_at).toLocaleString()}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">never</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={busyId === ch.id || !onTest}
                      onClick={() => handleTest(ch.id)}
                      data-testid={`channel-test-${ch.id}`}
                    >
                      {busyId === ch.id ? "Testing…" : "Test"}
                    </Button>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
