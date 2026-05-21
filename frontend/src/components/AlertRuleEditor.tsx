"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type {
  AlertRule,
  AlertRuleUpdate,
  AlertSeverity,
  AlertThresholdUnit,
  NotificationChannel,
} from "@/lib/types";

const SEVERITY_OPTIONS: AlertSeverity[] = ["critical", "warning", "info"];
const SEVERITY_LABELS: Record<AlertSeverity, string> = {
  critical: "Critical",
  warning: "Warning",
  info: "Info",
};

const MIN_WINDOW_SECONDS = 30;
const MAX_WINDOW_SECONDS = 24 * 60 * 60;

const SECONDS_MIN = 1;
const SECONDS_MAX = 24 * 60 * 60;

/**
 * Parse a free-form bytes string ("1024", "16 KB", "2.5MB") into an integer
 * count of bytes. Returns NaN if the input is not parseable.
 */
export function parseBytes(input: string): number {
  if (!input) return NaN;
  const cleaned = input.trim().replace(/,/g, "");
  const match = cleaned.match(/^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb|tb)?$/i);
  if (!match) return NaN;
  const value = parseFloat(match[1]);
  if (!Number.isFinite(value) || value < 0) return NaN;
  const unit = (match[2] || "b").toLowerCase();
  const multipliers: Record<string, number> = {
    b: 1,
    kb: 1024,
    mb: 1024 ** 2,
    gb: 1024 ** 3,
    tb: 1024 ** 4,
  };
  return Math.floor(value * multipliers[unit]);
}

function formatBytesForInput(n: number): string {
  if (!Number.isFinite(n)) return "";
  if (n === 0) return "0";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(2)} KB`;
  return String(n);
}

export interface AlertRuleEditorProps {
  rule: AlertRule;
  channels?: NotificationChannel[];
  saving?: boolean;
  saveError?: string | null;
  onSubmit: (payload: AlertRuleUpdate) => Promise<void> | void;
}

interface DraftState {
  thresholdInput: string;
  evaluation_window_seconds: number;
  severity: AlertSeverity;
  enabled: boolean;
  notification_channels: string[];
}

function initialDraft(rule: AlertRule): DraftState {
  return {
    thresholdInput:
      rule.threshold_unit === "bytes"
        ? formatBytesForInput(rule.threshold_value)
        : String(rule.threshold_value),
    evaluation_window_seconds: rule.evaluation_window_seconds,
    severity: rule.severity,
    enabled: rule.enabled,
    notification_channels: [...rule.notification_channels],
  };
}

function thresholdInputType(unit: AlertThresholdUnit): "text" | "number" {
  // "bytes" uses free-form text ("16 MB"); everything else is numeric.
  return unit === "bytes" ? "text" : "number";
}

function parseThreshold(unit: AlertThresholdUnit, raw: string): number {
  if (unit === "bytes") return parseBytes(raw);
  const v = Number(raw);
  if (!Number.isFinite(v)) return NaN;
  if (unit === "seconds") {
    if (v < SECONDS_MIN || v > SECONDS_MAX) return NaN;
    return Math.floor(v);
  }
  if (unit === "percent") {
    if (v < 0 || v > 100) return NaN;
    return v;
  }
  // count
  if (v < 0) return NaN;
  return Math.floor(v);
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  return sa.every((v, i) => v === sb[i]);
}

export function AlertRuleEditor({
  rule,
  channels,
  saving,
  saveError,
  onSubmit,
}: AlertRuleEditorProps) {
  const [draft, setDraft] = useState<DraftState>(() => initialDraft(rule));

  // Reset the draft when the upstream rule changes (e.g. after refresh).
  useEffect(() => {
    setDraft(initialDraft(rule));
  }, [rule]);

  const parsedThreshold = useMemo( => parseThreshold(rule.threshold_unit, draft.thresholdInput),
    [rule.threshold_unit, draft.thresholdInput]
  );
  const thresholdValid = Number.isFinite(parsedThreshold);

  const windowValid =
    draft.evaluation_window_seconds >= MIN_WINDOW_SECONDS &&
    draft.evaluation_window_seconds <= MAX_WINDOW_SECONDS;

  const changed =
    parsedThreshold !== rule.threshold_value ||
    draft.evaluation_window_seconds !== rule.evaluation_window_seconds ||
    draft.severity !== rule.severity ||
    draft.enabled !== rule.enabled ||
    !arraysEqual(draft.notification_channels, rule.notification_channels);

  const valid = thresholdValid && windowValid;
  const canSave = valid && changed && !saving;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSave) return;
    const payload: AlertRuleUpdate = {};
    if (parsedThreshold !== rule.threshold_value) {
      payload.threshold_value = parsedThreshold;
    }
    if (draft.evaluation_window_seconds !== rule.evaluation_window_seconds) {
      payload.evaluation_window_seconds = draft.evaluation_window_seconds;
    }
    if (draft.severity !== rule.severity) payload.severity = draft.severity;
    if (draft.enabled !== rule.enabled) payload.enabled = draft.enabled;
    if (!arraysEqual(draft.notification_channels, rule.notification_channels)) {
      payload.notification_channels = [...draft.notification_channels];
    }
    void onSubmit(payload);
  }

  function toggleChannel(id: string) {
    setDraft((d) => {
      const next = d.notification_channels.includes(id)
        ? d.notification_channels.filter((cid) => cid !== id)
        : [...d.notification_channels, id];
      return { ...d, notification_channels: next };
    });
  }

  return (
    <form
      className="space-y-5"
      onSubmit={handleSubmit}
      data-testid="alert-rule-editor"
      noValidate
    >
      <div>
        <h3 className="text-sm font-semibold">Expression (read-only)</h3>
        <p className="text-xs text-muted-foreground">
          PromQL is managed in git via <code>helm/ocr-local/templates/prometheusrule.yaml</code>
          {" "}and cannot be edited from this UI.
        </p>
        <pre
          className="mt-1 overflow-x-auto rounded-md border border-border bg-muted/30 p-3 text-xs"
          data-testid="rule-expression"
        >
          {rule.expression}
        </pre>
      </div>

      <div>
        <label className="block text-sm font-medium" htmlFor="threshold-input">
          Threshold ({rule.threshold_unit})
        </label>
        <Input
          id="threshold-input"
          type={thresholdInputType(rule.threshold_unit)}
          value={draft.thresholdInput}
          onChange={(e) =>
            setDraft((d) => ({ ...d, thresholdInput: e.target.value }))
          }
          aria-invalid={!thresholdValid}
          data-testid="threshold-input"
          {...(rule.threshold_unit !== "bytes"
            ? { min: rule.threshold_unit === "percent" ? 0 : 0 }
            : {})}
        />
        {!thresholdValid ? (
          <p
            className="mt-1 text-xs text-destructive"
            role="alert"
            data-testid="threshold-error"
          >
            Invalid {rule.threshold_unit} value.
          </p>
        ) : null}
      </div>

      <div>
        <label className="block text-sm font-medium" htmlFor="window-input">
          Evaluation window: {draft.evaluation_window_seconds}s
        </label>
        <input
          id="window-input"
          type="range"
          min={MIN_WINDOW_SECONDS}
          max={MAX_WINDOW_SECONDS}
          step={MIN_WINDOW_SECONDS}
          value={draft.evaluation_window_seconds}
          onChange={(e) =>
            setDraft((d) => ({
              ...d,
              evaluation_window_seconds: Number(e.target.value),
            }))
          }
          className="mt-1 w-full"
          data-testid="window-input"
        />
        <p className="text-xs text-muted-foreground">
          Range: {MIN_WINDOW_SECONDS}s – {MAX_WINDOW_SECONDS}s
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium" htmlFor="severity-select">
          Severity
        </label>
        <select
          id="severity-select"
          className="mt-1 block w-full rounded-md border border-input bg-background p-2 text-sm"
          value={draft.severity}
          onChange={(e) =>
            setDraft((d) => ({ ...d, severity: e.target.value as AlertSeverity }))
          }
          data-testid="severity-select"
        >
          {SEVERITY_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {SEVERITY_LABELS[s]}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={draft.enabled}
            onChange={(e) => setDraft((d) => ({ ...d, enabled: e.target.checked }))}
            data-testid="enabled-checkbox"
          />
          Enabled
        </label>
      </div>

      {channels && channels.length > 0 ? (
        <fieldset className="space-y-2" data-testid="channels-fieldset">
          <legend className="text-sm font-medium">Notification channels</legend>
          {channels.map((ch) => (
            <label
              key={ch.id}
              className="flex items-center gap-2 text-sm"
              data-testid={`channel-toggle-${ch.id}`}
            >
              <input
                type="checkbox"
                checked={draft.notification_channels.includes(ch.id)}
                onChange={() => toggleChannel(ch.id)}
              />
              <span className="font-mono text-xs">
                {ch.type}:{ch.target}
              </span>
              {!ch.enabled ? (
                <span className="text-xs text-muted-foreground">(disabled)</span>
              ) : null}
            </label>
          ))}
        </fieldset>
      ) : null}

      {saveError ? (
        <p
          className="text-sm text-destructive"
          role="alert"
          data-testid="rule-save-error"
        >
          {saveError}
        </p>
      ) : null}

      <div className="flex items-center gap-3">
        <Button
          type="submit"
          disabled={!canSave}
          data-testid="rule-save-button"
        >
          {saving ? "Saving…" : "Save changes"}
        </Button>
        {!changed ? (
          <span className="text-xs text-muted-foreground">No changes</span>
        ) : null}
      </div>
    </form>
  );
}
