"use client";

/**
 * D10 -- Tabbed settings form.
 *
 * Renders four <form>s -- one per tab -- each with its own Save button. The
 * tab strip is inlined here (rather than reusing the shared <Tabs> component)
 * because each tab needs to be its own <form> element with an independent
 * submit handler.
 */

import { useEffect, useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cn";
import {
  buildApiKeyRedactedPreview,
  isValidApiBaseUrl,
  listSupportedTimezones,
} from "@/lib/settings-store";
import { getApiKey } from "@/lib/auth";
import type {
  Settings,
  SettingsDateFormat,
  SettingsPageSize,
  SettingsTheme,
} from "@/lib/types";

type TabId = "general" | "api" | "display" | "notifications";

interface DeepPartialSettingsHelper {
  general?: Partial<Settings["general"]>;
  api?: Partial<Settings["api"]>;
  display?: Partial<Settings["display"]>;
  notifications?: Partial<Settings["notifications"]>;
}

export interface SettingsFormProps {
  settings: Settings;
  onUpdate: (patch: DeepPartialSettingsHelper) => void;
  onSave: => boolean;
  lastSavedAt: number | null;
}

const TAB_DEFS: { id: TabId; label: string }[] = [
  { id: "general", label: "General" },
  { id: "api", label: "API" },
  { id: "display", label: "Display" },
  { id: "notifications", label: "Notifications" },
];

const PAGE_SIZE_OPTIONS: SettingsPageSize[] = [10, 25, 50, 100];
const THEME_OPTIONS: SettingsTheme[] = ["light", "dark", "system"];
const DATE_FORMAT_OPTIONS: SettingsDateFormat[] = ["iso", "us", "eu"];

function formatRelativeSecondsAgo(then: number, now: number): string {
  const seconds = Math.max(0, Math.floor((now - then) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

export function SettingsForm({
  settings,
  onUpdate,
  onSave,
  lastSavedAt,
}: SettingsFormProps) {
  const [activeTab, setActiveTab] = useState<TabId>("general");
  const [tabSaveError, setTabSaveError] = useState<string | null>(null);
  const [tabJustSaved, setTabJustSaved] = useState<TabId | null>(null);
  const [, setTickNow] = useState<number>(Date.now());

  // Re-render once a second so the "Last saved Xs ago" label stays fresh.
  useEffect(() => {
    const id = setInterval(() => setTickNow(Date.now()), 1000);
    return => clearInterval(id);
  }, []);

  function attemptSave(tab: TabId, evt: FormEvent<HTMLFormElement>): void {
    evt.preventDefault();
    setTabSaveError(null);
    const ok = onSave();
    if (ok) {
      setTabJustSaved(tab);
    } else {
      setTabSaveError("Could not save settings. Browser storage may be full.");
    }
  }

  return (
    <div className="space-y-6">
      <div role="tablist" aria-label="Settings sections" className="flex gap-1 border-b border-border">
        {TAB_DEFS.map((t) => {
          const isActive = t.id === activeTab;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              data-testid={`settings-tab-${t.id}`}
              className={cn(
                "px-3 py-2 text-sm font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                isActive
                  ? "border-b-2 border-primary text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
              onClick={() => {
                setActiveTab(t.id);
                setTabSaveError(null);
              }}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {activeTab === "general" && (
        <GeneralPanel
          settings={settings}
          onUpdate={onUpdate}
          onSubmit={(e) => attemptSave("general", e)}
          saveError={tabSaveError}
          justSaved={tabJustSaved === "general"}
          lastSavedAt={lastSavedAt}
        />
      )}

      {activeTab === "api" && (
        <ApiPanel
          settings={settings}
          onUpdate={onUpdate}
          onSubmit={(e) => attemptSave("api", e)}
          saveError={tabSaveError}
          justSaved={tabJustSaved === "api"}
          lastSavedAt={lastSavedAt}
        />
      )}

      {activeTab === "display" && (
        <DisplayPanel
          settings={settings}
          onUpdate={onUpdate}
          onSubmit={(e) => attemptSave("display", e)}
          saveError={tabSaveError}
          justSaved={tabJustSaved === "display"}
          lastSavedAt={lastSavedAt}
        />
      )}

      {activeTab === "notifications" && (
        <NotificationsPanel
          settings={settings}
          onUpdate={onUpdate}
          onSubmit={(e) => attemptSave("notifications", e)}
          saveError={tabSaveError}
          justSaved={tabJustSaved === "notifications"}
          lastSavedAt={lastSavedAt}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel: General
// ---------------------------------------------------------------------------

interface PanelProps {
  settings: Settings;
  onUpdate: (patch: DeepPartialSettingsHelper) => void;
  onSubmit: (evt: FormEvent<HTMLFormElement>) => void;
  saveError: string | null;
  justSaved: boolean;
  lastSavedAt: number | null;
}

function SaveBar({ justSaved, lastSavedAt, error }: {
  justSaved: boolean;
  lastSavedAt: number | null;
  error: string | null;
}) {
  return (
    <div className="flex items-center gap-3">
      <Button type="submit" data-testid="settings-save-button">
        Save
      </Button>
      {justSaved && lastSavedAt !== null && (
        <span className="text-xs text-muted-foreground" data-testid="settings-last-saved">
          Last saved {formatRelativeSecondsAgo(lastSavedAt, Date.now())}
        </span>
      )}
      {error && (
        <span className="text-xs text-destructive" data-testid="settings-save-error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}

function GeneralPanel({ settings, onUpdate, onSubmit, saveError, justSaved, lastSavedAt }: PanelProps) {
  const timezones = listSupportedTimezones();

  return (
    <form onSubmit={onSubmit} className="space-y-5" data-testid="settings-form-general">
      <div className="space-y-2">
        <label className="block text-sm font-medium" htmlFor="settings-theme">Theme</label>
        <div className="flex gap-2" role="radiogroup" aria-label="Theme">
          {THEME_OPTIONS.map((t) => (
            <button
              key={t}
              type="button"
              role="radio"
              aria-checked={settings.general.theme === t}
              data-testid={`settings-theme-${t}`}
              onClick={() => onUpdate({ general: { theme: t } })}
              className={cn(
                "rounded-md border px-3 py-1.5 text-sm capitalize transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                settings.general.theme === t
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-input bg-background hover:bg-accent hover:text-accent-foreground"
              )}
            >
              {t}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Theme applies immediately. Save persists the choice across reloads.
        </p>
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium" htmlFor="settings-timezone">
          Timezone
        </label>
        <select
          id="settings-timezone"
          data-testid="settings-timezone"
          value={settings.general.timezone}
          onChange={(e) => onUpdate({ general: { timezone: e.target.value } })}
          className="block w-full max-w-md rounded-md border border-input bg-background px-3 py-2 text-sm"
        >
          {/* Make sure the current value is selectable even if it isn't in the list. */}
          {!timezones.includes(settings.general.timezone) && (
            <option value={settings.general.timezone}>{settings.general.timezone}</option>
          )}
          {timezones.map((tz) => (
            <option key={tz} value={tz}>
              {tz}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium" htmlFor="settings-date-format">
          Date format
        </label>
        <select
          id="settings-date-format"
          data-testid="settings-date-format"
          value={settings.general.dateFormat}
          onChange={(e) =>
            onUpdate({ general: { dateFormat: e.target.value as SettingsDateFormat } })
          }
          className="block w-40 rounded-md border border-input bg-background px-3 py-2 text-sm"
        >
          {DATE_FORMAT_OPTIONS.map((f) => (
            <option key={f} value={f}>
              {f.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      <SaveBar justSaved={justSaved} lastSavedAt={lastSavedAt} error={saveError} />
    </form>
  );
}

// ---------------------------------------------------------------------------
// Panel: API
// ---------------------------------------------------------------------------

function ApiPanel({ settings, onUpdate, onSubmit, saveError, justSaved, lastSavedAt }: PanelProps) {
  const [baseUrlError, setBaseUrlError] = useState<string | null>(null);
  const [timeoutError, setTimeoutError] = useState<string | null>(null);

  function handleSync() {
    const key = getApiKey();
    onUpdate({ api: { apiKeyRedactedPreview: buildApiKeyRedactedPreview(key) } });
  }

  function submit(evt: FormEvent<HTMLFormElement>) {
    setBaseUrlError(null);
    setTimeoutError(null);
    if (!isValidApiBaseUrl(settings.api.baseUrl)) {
      setBaseUrlError("Must be a valid URL or empty (current origin).");
      evt.preventDefault();
      return;
    }
    if (
      !Number.isFinite(settings.api.timeoutMs) ||
      settings.api.timeoutMs < 1000 ||
      settings.api.timeoutMs > 120000
    ) {
      setTimeoutError("Timeout must be between 1000 and 120000 ms.");
      evt.preventDefault();
      return;
    }
    onSubmit(evt);
  }

  return (
    <form onSubmit={submit} className="space-y-5" data-testid="settings-form-api">
      <div className="space-y-2">
        <label className="block text-sm font-medium" htmlFor="settings-base-url">
          API base URL
        </label>
        <Input
          id="settings-base-url"
          data-testid="settings-base-url"
          value={settings.api.baseUrl}
          placeholder="https://ocr.example.com (empty = current origin)"
          onChange={(e) => onUpdate({ api: { baseUrl: e.target.value } })}
        />
        {baseUrlError && (
          <p className="text-xs text-destructive" data-testid="settings-base-url-error">
            {baseUrlError}
          </p>
        )}
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium" htmlFor="settings-timeout-ms">
          Request timeout (ms)
        </label>
        <Input
          id="settings-timeout-ms"
          data-testid="settings-timeout-ms"
          type="number"
          min={1000}
          max={120000}
          step={500}
          value={String(settings.api.timeoutMs)}
          onChange={(e) =>
            onUpdate({ api: { timeoutMs: Number.parseInt(e.target.value, 10) || 0 } })
          }
        />
        {timeoutError && (
          <p className="text-xs text-destructive" data-testid="settings-timeout-error">
            {timeoutError}
          </p>
        )}
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium">API key (preview)</label>
        <div className="flex items-center gap-3">
          <code
            data-testid="settings-api-key-preview"
            className="rounded bg-muted px-2 py-1 text-xs font-mono"
          >
            {settings.api.apiKeyRedactedPreview || "(not set)"}
          </code>
          <Button
            type="button"
            variant="outline"
            size="sm"
            data-testid="settings-api-key-sync"
            onClick={handleSync}
          >
            Sync from auth
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          The actual key is owned by the auth layer; only a redacted preview is stored here.
        </p>
      </div>

      <SaveBar justSaved={justSaved} lastSavedAt={lastSavedAt} error={saveError} />
    </form>
  );
}

// ---------------------------------------------------------------------------
// Panel: Display
// ---------------------------------------------------------------------------

function DisplayPanel({ settings, onUpdate, onSubmit, saveError, justSaved, lastSavedAt }: PanelProps) {
  return (
    <form onSubmit={onSubmit} className="space-y-5" data-testid="settings-form-display">
      <div className="space-y-2">
        <label className="block text-sm font-medium" htmlFor="settings-auto-refresh">
          Auto-refresh interval (seconds): {settings.display.autoRefreshSeconds}
        </label>
        <input
          id="settings-auto-refresh"
          data-testid="settings-auto-refresh"
          type="range"
          min={5}
          max={600}
          step={5}
          value={settings.display.autoRefreshSeconds}
          onChange={(e) =>
            onUpdate({ display: { autoRefreshSeconds: Number.parseInt(e.target.value, 10) } })
          }
          className="block w-full max-w-md"
        />
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium" htmlFor="settings-page-size">
          Page size
        </label>
        <select
          id="settings-page-size"
          data-testid="settings-page-size"
          value={settings.display.pageSize}
          onChange={(e) =>
            onUpdate({
              display: { pageSize: Number.parseInt(e.target.value, 10) as SettingsPageSize },
            })
          }
          className="block w-32 rounded-md border border-input bg-background px-3 py-2 text-sm"
        >
          {PAGE_SIZE_OPTIONS.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>

      <ToggleRow
        id="settings-compact-mode"
        label="Compact mode"
        description="Reduce row padding in tables for denser displays."
        checked={settings.display.compactMode}
        onChange={(v) => onUpdate({ display: { compactMode: v } })}
      />

      <ToggleRow
        id="settings-advanced-columns"
        label="Show advanced columns"
        description="Reveal extra fields (worker id, queue, retry counts) in tables."
        checked={settings.display.showAdvancedColumns}
        onChange={(v) => onUpdate({ display: { showAdvancedColumns: v } })}
      />

      <SaveBar justSaved={justSaved} lastSavedAt={lastSavedAt} error={saveError} />
    </form>
  );
}

// ---------------------------------------------------------------------------
// Panel: Notifications
// ---------------------------------------------------------------------------

function NotificationsPanel({
  settings,
  onUpdate,
  onSubmit,
  saveError,
  justSaved,
  lastSavedAt,
}: PanelProps) {
  const [permState, setPermState] = useState<NotificationPermission | "unsupported">(() => {
    if (typeof window === "undefined" || typeof Notification === "undefined") {
      return "unsupported";
    }
    return Notification.permission;
  });

  async function requestPerm() {
    if (typeof window === "undefined" || typeof Notification === "undefined") {
      return;
    }
    try {
      const next = await Notification.requestPermission();
      setPermState(next);
      if (next !== "granted") {
        // If user denied, surface the reality back into settings.
        onUpdate({ notifications: { desktopEnabled: false } });
      }
    } catch {
      setPermState("denied");
    }
  }

  const desktopAvailable = permState !== "unsupported" && permState !== "denied";
  const subTogglesDisabled = !settings.notifications.desktopEnabled;

  return (
    <form onSubmit={onSubmit} className="space-y-5" data-testid="settings-form-notifications">
      <div className="space-y-2">
        <ToggleRow
          id="settings-notif-desktop"
          label="Desktop notifications"
          description={
            permState === "unsupported"
              ? "Your browser does not support the Notification API."
              : permState === "denied"
                ? "Notifications were denied. Adjust the permission in your browser settings."
                : permState === "granted"
                  ? "Notifications permission granted."
                  : "Enable to request browser permission."
          }
          checked={settings.notifications.desktopEnabled && desktopAvailable}
          disabled={!desktopAvailable}
          onChange={(v) => {
            if (v && permState === "default") {
              void requestPerm();
            }
            onUpdate({ notifications: { desktopEnabled: v && desktopAvailable } });
          }}
        />
        {permState === "default" && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            data-testid="settings-notif-request-permission"
            onClick={() => void requestPerm()}
          >
            Request permission
          </Button>
        )}
      </div>

      <ToggleRow
        id="settings-notif-sound"
        label="Sound"
        description="Play a short tone when a notification fires."
        checked={settings.notifications.soundEnabled}
        disabled={subTogglesDisabled}
        onChange={(v) => onUpdate({ notifications: { soundEnabled: v } })}
      />

      <ToggleRow
        id="settings-notif-on-job-complete"
        label="On job complete"
        description="Notify when an OCR job finishes successfully."
        checked={settings.notifications.onJobComplete}
        disabled={subTogglesDisabled}
        onChange={(v) => onUpdate({ notifications: { onJobComplete: v } })}
      />

      <ToggleRow
        id="settings-notif-on-job-failure"
        label="On job failure"
        description="Notify when an OCR job fails."
        checked={settings.notifications.onJobFailure}
        disabled={subTogglesDisabled}
        onChange={(v) => onUpdate({ notifications: { onJobFailure: v } })}
      />

      <ToggleRow
        id="settings-notif-on-review-item"
        label="On review item"
        description="Notify when a new review queue item arrives."
        checked={settings.notifications.onReviewItem}
        disabled={subTogglesDisabled}
        onChange={(v) => onUpdate({ notifications: { onReviewItem: v } })}
      />

      <SaveBar justSaved={justSaved} lastSavedAt={lastSavedAt} error={saveError} />
    </form>
  );
}

// ---------------------------------------------------------------------------
// Toggle row primitive
// ---------------------------------------------------------------------------

interface ToggleRowProps {
  id: string;
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}

function ToggleRow({ id, label, description, checked, onChange, disabled = false }: ToggleRowProps) {
  return (
    <label
      htmlFor={id}
      className={cn(
        "flex items-start gap-3",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <input
        id={id}
        data-testid={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 rounded border-input"
      />
      <span>
        <span className="block text-sm font-medium">{label}</span>
        <span className="block text-xs text-muted-foreground">{description}</span>
      </span>
    </label>
  );
}
