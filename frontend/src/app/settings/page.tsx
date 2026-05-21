"use client";

/**
 * D10 -- /settings page.
 *
 * Hosts the SettingsForm controller plus the page-level "Reset to defaults"
 * button, which uses two-click confirmation to avoid accidental wipes.
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { SettingsForm } from "@/components/SettingsForm";
import { useRequireAuth } from "@/lib/auth";
import { useSettings } from "@/lib/hooks";

export default function SettingsPage() {
  useRequireAuth();
  const { settings, update, save, reset, lastSavedAt } = useSettings();
  const [confirmingReset, setConfirmingReset] = useState<boolean>(false);

  function handleResetClick() {
    if (!confirmingReset) {
      setConfirmingReset(true);
      return;
    }
    reset();
    setConfirmingReset(false);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
          <p className="text-sm text-muted-foreground">
            Browser-local preferences. Stored under{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">
              ocr-local:settings
            </code>{" "}
            in this browser only.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {confirmingReset && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              data-testid="settings-reset-cancel"
              onClick={() => setConfirmingReset(false)}
            >
              Cancel
            </Button>
          )}
          <Button
            type="button"
            variant={confirmingReset ? "destructive" : "outline"}
            size="sm"
            data-testid="settings-reset-button"
            onClick={handleResetClick}
          >
            {confirmingReset ? "Confirm reset" : "Reset to defaults"}
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Preferences</CardTitle>
          <CardDescription>
            Each section saves independently. Theme changes apply immediately;
            other changes apply after Save.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <SettingsForm
            settings={settings}
            onUpdate={update}
            onSave={save}
            lastSavedAt={lastSavedAt}
          />
        </CardContent>
      </Card>
    </div>
  );
}
