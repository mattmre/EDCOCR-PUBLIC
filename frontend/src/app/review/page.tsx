"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { Pagination } from "@/components/Pagination";
import {
  EMPTY_REVIEW_FILTER_STATE,
  ReviewQueueFiltersBar,
} from "@/components/ReviewQueueFilters";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { useRequireAuth } from "@/lib/auth";
import { useReviewQueue } from "@/lib/hooks";
import type { ReviewQueueFilters, ReviewStatus } from "@/lib/types";

const DEFAULT_PAGE_SIZE = 25;

const VALID_STATUSES: ReviewStatus[] = [
  "pending",
  "approved",
  "rejected",
  "reprocess",
];

function parseFiltersFromUrl(params: URLSearchParams): ReviewQueueFilters {
  const rawStatuses = params.getAll("status") as string[];
  const statuses = rawStatuses.filter((s): s is ReviewStatus =>
    (VALID_STATUSES as string[]).includes(s)
  );
  return {
    status: statuses.length > 0 ? statuses : ["pending"],
    reason: params.get("reason") ?? "",
    q: params.get("q") ?? "",
  };
}

function filtersToUrl(
  filters: ReviewQueueFilters,
  page: number
): string {
  const out = new URLSearchParams();
  for (const s of filters.status) out.append("status", s);
  if (filters.reason) out.set("reason", filters.reason);
  if (filters.q) out.set("q", filters.q);
  if (page > 1) out.set("page", String(page));
  const qs = out.toString();
  return qs ? `?${qs}` : "";
}

function formatTimestamp(value: string): string {
  if (!value) return "—";
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    return d.toLocaleString();
  } catch {
    return value;
  }
}

function relativeTime(iso: string, nowMs: number): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const deltaSec = Math.max(0, Math.floor((nowMs - t) / 1000));
  if (deltaSec < 5) return "just now";
  if (deltaSec < 60) return `${deltaSec}s ago`;
  if (deltaSec < 3600) return `${Math.floor(deltaSec / 60)}m ago`;
  if (deltaSec < 86400) return `${Math.floor(deltaSec / 3600)}h ago`;
  return `${Math.floor(deltaSec / 86400)}d ago`;
}

function ReviewQueueContent() {
  useRequireAuth();
  const searchParams = useSearchParams();
  const router = useRouter();

  const initialFilters = useMemo( => parseFiltersFromUrl(searchParams ?? new URLSearchParams()),
    // Only parse once on mount; subsequent updates are managed in component state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );
  const initialPage = useMemo(() => {
    const raw = searchParams?.get("page");
    const n = raw ? Number(raw) : 1;
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : 1;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [filters, setFilters] = useState<ReviewQueueFilters>(initialFilters);
  const [page, setPage] = useState<number>(initialPage);

  const offset = (page - 1) * DEFAULT_PAGE_SIZE;

  const queueQuery = useMemo( => ({ ...filters, limit: DEFAULT_PAGE_SIZE, offset }),
    [filters, offset]
  );

  const { data, error, loading, refresh, lastUpdated } = useReviewQueue(queueQuery);

  // Keep the URL in sync with state for shareable links.
  useEffect(() => {
    const url = `/review${filtersToUrl(filters, page)}`;
    router.replace(url);
  }, [filters, page, router]);

  // When filters change, reset to first page.
  const handleFiltersChange = useCallback((next: ReviewQueueFilters) => {
    setFilters(next);
    setPage(1);
  }, []);

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const showLoading = loading && items.length === 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Review</h1>
        <p className="text-sm text-muted-foreground">
          Human review queue. Auto-refreshes every 30 seconds.
        </p>
      </div>

      <ReviewQueueFiltersBar value={filters} onChange={handleFiltersChange} />

      {error ? (
        <div
          role="alert"
          data-testid="review-error"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
        >
          Failed to load review queue: {error.message}
          <button
            type="button"
            className="ml-3 text-xs underline"
            onClick={() => refresh()}
          >
            Retry
          </button>
        </div>
      ) : null}

      <div className="overflow-x-auto rounded-md border border-border">
        <table
          className="min-w-full divide-y divide-border"
          data-testid="review-table"
          aria-label="Review queue items"
        >
          <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
            <tr>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Review ID
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Job ID
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Reason
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Confidence
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Submitted
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Reviewer
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Status
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border bg-background text-sm">
            {showLoading ? (
              <tr>
                <td
                  colSpan={8}
                  className="px-3 py-6 text-center text-muted-foreground"
                  data-testid="review-loading"
                >
                  Loading review queue…
                </td>
              </tr>
            ) : items.length === 0 ? (
              <tr>
                <td
                  colSpan={8}
                  className="px-3 py-6 text-center text-muted-foreground"
                  data-testid="review-empty"
                >
                  No review items match the current filters.
                </td>
              </tr>
            ) : (
              items.map((item) => {
                const conf = `${(item.confidence * 100).toFixed(1)}%`;
                const now = lastUpdated ?? Date.now();
                return (
                  <tr
                    key={item.review_id}
                    className="cursor-pointer hover:bg-muted/30"
                    data-testid={`review-row-${item.review_id}`}
                    onClick={() => router.push(`/review/${encodeURIComponent(item.review_id)}`)}
                  >
                    <td className="px-3 py-2 font-mono text-xs">
                      <Link
                        href={`/review/${encodeURIComponent(item.review_id)}`}
                        className="text-primary hover:underline"
                        data-testid={`review-link-${item.review_id}`}
                        onClick={(e) => e.stopPropagation()}
                      >
                        {item.review_id}
                      </Link>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                      {item.job_id}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {item.reason || "—"}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{conf}</td>
                    <td
                      className="px-3 py-2 text-muted-foreground"
                      title={formatTimestamp(item.created_at)}
                    >
                      {relativeTime(item.created_at, now)}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {item.reviewer || "—"}
                    </td>
                    <td className="px-3 py-2">
                      <StatusBadge status={item.status} />
                    </td>
                    <td className="px-3 py-2">
                      <Link
                        href={`/review/${encodeURIComponent(item.review_id)}`}
                        className="text-xs text-primary hover:underline"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Open
                      </Link>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <Pagination
        page={page}
        pageSize={DEFAULT_PAGE_SIZE}
        total={total}
        onPageChange={setPage}
        disabled={loading}
      />
    </div>
  );
}

export default function ReviewQueuePage() {
  return (
    <Suspense
      fallback={
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-semibold">Review</h1>
            <p className="text-sm text-muted-foreground">
              Human review queue. Auto-refreshes every 30 seconds.
            </p>
          </div>
          <div
            className="rounded-md border border-border p-6 text-sm text-muted-foreground"
            data-testid="review-loading"
          >
            Loading review queue...
          </div>
        </div>
      }
    >
      <ReviewQueueContent />
    </Suspense>
  );
}
