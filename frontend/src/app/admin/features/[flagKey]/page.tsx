"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FeatureFlagDetail } from "@/components/FeatureFlagDetail";
import { ApiError, UnauthorizedError } from "@/lib/api-client";
import { useRequireAuth } from "@/lib/auth";
import { useFeatureFlag, useFlagHistory } from "@/lib/hooks";

interface EmptyStateProps {
  title: string;
  body: string;
}

function EmptyState({ title, body }: EmptyStateProps) {
  return (
    <Card data-testid="flag-detail-empty-state">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{body}</CardDescription>
      </CardHeader>
      <CardContent>
        <Link
          href="/admin/features"
          className="text-xs text-primary hover:underline"
        >
          Back to feature flags
        </Link>
      </CardContent>
    </Card>
  );
}

export default function FeatureFlagDetailPage() {
  useRequireAuth();
  const params = useParams<{ flagKey: string }>();
  const flagKey = decodeURIComponent((params?.flagKey ?? "") as string);

  const flag = useFeatureFlag(flagKey);
  const history = useFlagHistory(flagKey);

  const errorState = useMemo(() => {
    if (!flag.error) return null;
    if (
      flag.error instanceof UnauthorizedError ||
      (flag.error instanceof ApiError && flag.error.status === 403)
    ) {
      return (
        <EmptyState
          title="Platform admin role required"
          body="Flag detail and change requests require platform-admin scope."
        />
      );
    }
    // 404 surfaces as `data === null` (the API client maps it).
    return (
      <Card data-testid="flag-detail-error">
        <CardHeader>
          <CardTitle>Failed to load flag</CardTitle>
          <CardDescription>{flag.error.message}</CardDescription>
        </CardHeader>
      </Card>
    );
  }, [flag.error]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <Link
            href="/admin/features"
            className="text-xs text-primary hover:underline"
            data-testid="flag-detail-back"
          >
            ← Back to feature flags
          </Link>
          <h1 className="mt-2 text-2xl font-semibold">Feature flag</h1>
        </div>
      </div>

      {errorState ? (
        errorState
      ) : flag.loading && !flag.data ? (
        <p className="text-sm text-muted-foreground">Loading flag…</p>
      ) : !flag.data ? (
        <EmptyState
          title="Flag not found"
          body={`No feature flag with key '${flagKey}' is registered.`}
        />
      ) : (
        <FeatureFlagDetail
          flag={flag.data}
          history={history.data}
          historyLoading={history.loading}
          historyError={history.error}
          onChangeRequested={() => {
            flag.refresh();
            history.refresh();
          }}
        />
      )}
    </div>
  );
}
