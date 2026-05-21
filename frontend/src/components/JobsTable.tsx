"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Pagination } from "@/components/Pagination";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  EMPTY_FILTER_STATE,
  JobsFilterBar,
  jobsFilterToQuery,
} from "@/components/JobsFilterBar";
import { ApiError, get } from "@/lib/api-client";
import type { Job, JobListResponse, JobsFilterState } from "@/lib/types";

const DEFAULT_PAGE_SIZE = 25;
const REFRESH_INTERVAL_MS = 10_000;

interface State {
  jobs: Job[];
  total: number;
  loading: boolean;
  error: string | null;
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    return d.toLocaleString();
  } catch {
    return value;
  }
}

function durationSeconds(job: Job): string {
  if (job.completed_at && job.started_at) {
    const diff =
      (new Date(job.completed_at).getTime() -
        new Date(job.started_at).getTime()) /
      1000;
    if (Number.isFinite(diff) && diff >= 0) return `${diff.toFixed(1)}s`;
  }
  return "—";
}

export interface JobsTableProps {
  /** Override the default page size (mainly for tests). */
  pageSize?: number;
  /** Disable polling (for unit tests). */
  refresh?: boolean;
  /** Initial filter state -- defaults to EMPTY_FILTER_STATE. */
  initialFilters?: JobsFilterState;
}

export function JobsTable({
  pageSize = DEFAULT_PAGE_SIZE,
  refresh = true,
  initialFilters = EMPTY_FILTER_STATE,
}: JobsTableProps) {
  const [filters, setFilters] = useState<JobsFilterState>(initialFilters);
  const [page, setPage] = useState<number>(1);

  const [state, setState] = useState<State>({
    jobs: [],
    total: 0,
    loading: true,
    error: null,
  });

  const filtersChangingRef = useRef<boolean>(false);
  const offset = (page - 1) * pageSize;

  // When filters change, reset to first page.
  useEffect(() => {
    setPage(1);
  }, [filters]);

  // Initial fetch + polling.
  useEffect(() => {
    void load(true);
    if (!refresh) return;
    const id = setInterval(() => {
      if (filtersChangingRef.current) return;
      void load(false);
    }, REFRESH_INTERVAL_MS);
    return => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh, filters, page, pageSize]);

  async function load(showLoading: boolean) {
    if (showLoading) setState((s) => ({ ...s, loading: true, error: null }));
    const params = jobsFilterToQuery(filters);
    params.set("limit", String(pageSize));
    params.set("offset", String(offset));
    try {
      const response = await get<JobListResponse>(
        `/api/v1/jobs?${params.toString()}`
      );
      setState({
        jobs: response.jobs ?? [],
        total: response.total ?? 0,
        loading: false,
        error: null,
      });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? `${err.status} ${err.message}`
          : err instanceof Error
            ? err.message
            : "unknown error";
      setState({ jobs: [], total: 0, loading: false, error: message });
    }
  }

  function handleFiltersChange(next: JobsFilterState) {
    filtersChangingRef.current = true;
    setFilters(next);
    // Release the polling-pause flag on the next tick so a manual fetch can run.
    setTimeout(() => {
      filtersChangingRef.current = false;
    }, 0);
  }

  return (
    <div className="space-y-4">
      <JobsFilterBar value={filters} onChange={handleFiltersChange} />

      {state.error ? (
        <div
          data-testid="jobs-error"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
        >
          Failed to load jobs: {state.error}
          <Button
            variant="outline"
            size="sm"
            className="ml-3"
            onClick={() => void load(true)}
          >
            Retry
          </Button>
        </div>
      ) : null}

      <div className="overflow-x-auto rounded-md border border-border">
        <table className="min-w-full divide-y divide-border" data-testid="jobs-table">
          <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
            <tr>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Job ID
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Status
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Source
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Created
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Updated
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Duration
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Pages
              </th>
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Progress
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border bg-background text-sm">
            {state.loading && state.jobs.length === 0 ? (
              <tr>
                <td
                  colSpan={8}
                  className="px-3 py-6 text-center text-muted-foreground"
                  data-testid="jobs-loading"
                >
                  Loading jobs…
                </td>
              </tr>
            ) : state.jobs.length === 0 ? (
              <tr>
                <td
                  colSpan={8}
                  className="px-3 py-6 text-center text-muted-foreground"
                  data-testid="jobs-empty"
                >
                  No jobs found
                </td>
              </tr>
            ) : (
              state.jobs.map((job) => {
                const progress = job.progress;
                const updated = job.completed_at ?? job.started_at ?? job.created_at;
                return (
                  <tr key={job.job_id} className="hover:bg-muted/30">
                    <td className="px-3 py-2 font-mono text-xs">
                      <Link
                        href={`/jobs/${job.job_id}`}
                        className="text-primary hover:underline"
                        data-testid={`job-link-${job.job_id}`}
                      >
                        {job.job_id}
                      </Link>
                    </td>
                    <td className="px-3 py-2">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-3 py-2 max-w-xs truncate" title={job.source_file}>
                      {job.source_file || "—"}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {formatTimestamp(job.created_at)}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {formatTimestamp(updated)}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{durationSeconds(job)}</td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {progress
                        ? `${progress.pages_completed}/${progress.total_pages}`
                        : "—"}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {progress ? `${progress.percent_complete.toFixed(0)}%` : "—"}
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
        pageSize={pageSize}
        total={state.total}
        onPageChange={setPage}
        disabled={state.loading}
      />
    </div>
  );
}
