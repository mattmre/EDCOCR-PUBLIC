"use client";

import { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FeatureFlagsList } from "@/components/FeatureFlagsList";
import { ApiError, UnauthorizedError } from "@/lib/api-client";
import { useRequireAuth } from "@/lib/auth";
import { useFeatureFlags } from "@/lib/hooks";

interface EmptyStateProps {
  title: string;
  body: string;
}

function EmptyState({ title, body }: EmptyStateProps) {
  return (
    <Card data-testid="flags-empty-state">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{body}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground">
          Until the backend admin surface is wired up, view the flag definitions
          directly at <code className="rounded bg-muted px-1">ocr_local/config/feature_flags.py</code>.
        </p>
      </CardContent>
    </Card>
  );
}

export default function FeatureFlagsListPage() {
  useRequireAuth();
  const { data, error, loading, lastUpdated, refresh } = useFeatureFlags();

  const errorState = useMemo(() => {
    if (!error) return null;
    if (
      error instanceof UnauthorizedError ||
      (error instanceof ApiError && error.status === 403)
    ) {
      return (
        <EmptyState
          title="Platform admin role required"
          body="The feature-flag admin surface requires platform-admin scope. Use a key with admin role to view this page."
        />
      );
    }
    if (
      error instanceof ApiError &&
      [404, 501].includes(error.status)
    ) {
      return (
        <EmptyState
          title="Feature flags API not yet provisioned"
          body="The backend has not exposed /api/v1/admin/feature-flags. View ocr_local/config/feature_flags.py directly until the admin surface is wired up."
        />
      );
    }
    return (
      <Card data-testid="flags-error">
        <CardHeader>
          <CardTitle>Failed to load flags</CardTitle>
          <CardDescription>{error.message}</CardDescription>
        </CardHeader>
      </Card>
    );
  }, [error]);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Feature Flags</h1>
          <p className="text-sm text-muted-foreground">
            Read-only inventory of pipeline, custody, translation, and
            operational feature flags. Changes are filed as custody-logged
            change requests; the backend retains the final authority on flips.
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          className="rounded-md border border-border px-3 py-1 text-xs hover:bg-muted/30"
          data-testid="flags-refresh"
        >
          Refresh
        </button>
      </div>

      {lastUpdated ? (
        <p className="text-xs text-muted-foreground">
          Last updated: {new Date(lastUpdated).toISOString()}
        </p>
      ) : null}

      {errorState ? (
        errorState
      ) : (
        <FeatureFlagsList flags={data ?? []} loading={loading} />
      )}
    </div>
  );
}
