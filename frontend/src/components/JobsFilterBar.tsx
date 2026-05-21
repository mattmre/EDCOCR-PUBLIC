"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { JobsFilterState, JobsSort } from "@/lib/types";

const STATUS_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "submitted", label: "Submitted" },
  { value: "queued", label: "Queued" },
  { value: "processing", label: "Processing" },
  { value: "running", label: "Running" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
  { value: "cancelled", label: "Cancelled" },
];

const SORT_OPTIONS: Array<{ value: JobsSort; label: string }> = [
  { value: "submitted_at_desc", label: "Newest first" },
  { value: "submitted_at_asc", label: "Oldest first" },
  { value: "duration_desc", label: "Longest duration" },
  { value: "status", label: "By status" },
];

const SEARCH_DEBOUNCE_MS = 300;

export const EMPTY_FILTER_STATE: JobsFilterState = {
  status: [],
  submitted_after: "",
  submitted_before: "",
  q: "",
  sort: "submitted_at_desc",
};

export interface JobsFilterBarProps {
  value: JobsFilterState;
  onChange: (next: JobsFilterState) => void;
  /** Debounce window for the free-text search box, in milliseconds. */
  searchDebounceMs?: number;
}

/**
 * D6 server-side jobs filter bar.
 *
 * - Status is a multi-select chip group; clicking a chip toggles it.
 * - Search has a 300 ms debounce so we don't fan out one fetch per keystroke.
 * - All other filters fire on change.
 */
export function JobsFilterBar({
  value,
  onChange,
  searchDebounceMs = SEARCH_DEBOUNCE_MS,
}: JobsFilterBarProps) {
  // Local state for the search input so we can debounce without lagging the textbox.
  const [searchDraft, setSearchDraft] = useState<string>(value.q);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync external resets (e.g. URL state) into the local draft.
  useEffect(() => {
    setSearchDraft(value.q);
  }, [value.q]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (searchDraft === value.q) return;
    debounceRef.current = setTimeout(() => {
      onChange({ ...value, q: searchDraft });
    }, searchDebounceMs);
    return => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchDraft, searchDebounceMs, value, onChange]);

  function toggleStatus(s: string) {
    const isOn = value.status.includes(s);
    const next = isOn
      ? value.status.filter((existing) => existing !== s)
      : [...value.status, s];
    onChange({ ...value, status: next });
  }

  function clearAll() {
    setSearchDraft("");
    onChange(EMPTY_FILTER_STATE);
  }

  const hasAnyFilter =
    value.status.length > 0 ||
    value.submitted_after !== "" ||
    value.submitted_before !== "" ||
    value.q !== "" ||
    value.sort !== "submitted_at_desc";

  return (
    <div
      className="space-y-3 rounded-md border border-border bg-background p-3"
      data-testid="jobs-filter-bar"
    >
      <div className="flex flex-wrap gap-2" role="group" aria-label="Status filter">
        {STATUS_OPTIONS.map((opt) => {
          const active = value.status.includes(opt.value);
          return (
            <button
              key={opt.value}
              type="button"
              data-testid={`status-chip-${opt.value}`}
              aria-pressed={active}
              onClick={() => toggleStatus(opt.value)}
              className={
                "rounded-full border px-3 py-1 text-xs font-medium transition-colors " +
                (active
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-input bg-background text-foreground hover:bg-muted")
              }
            >
              {opt.label}
            </button>
          );
        })}
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <label className="block text-xs font-medium text-muted-foreground">
          Search
          <Input
            data-testid="jobs-filter-search"
            type="search"
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            placeholder="Filename or job_id"
            className="mt-1"
            maxLength={256}
          />
        </label>
        <label className="block text-xs font-medium text-muted-foreground">
          Submitted after
          <Input
            data-testid="jobs-filter-submitted-after"
            type="datetime-local"
            value={value.submitted_after}
            onChange={(e) =>
              onChange({ ...value, submitted_after: e.target.value })
            }
            className="mt-1"
          />
        </label>
        <label className="block text-xs font-medium text-muted-foreground">
          Submitted before
          <Input
            data-testid="jobs-filter-submitted-before"
            type="datetime-local"
            value={value.submitted_before}
            onChange={(e) =>
              onChange({ ...value, submitted_before: e.target.value })
            }
            className="mt-1"
          />
        </label>
        <label className="block text-xs font-medium text-muted-foreground">
          Sort
          <select
            data-testid="jobs-filter-sort"
            value={value.sort}
            onChange={(e) =>
              onChange({ ...value, sort: e.target.value as JobsSort })
            }
            className="mt-1 block h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex items-center justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={clearAll}
          disabled={!hasAnyFilter}
          data-testid="jobs-filter-clear"
        >
          Clear filters
        </Button>
      </div>
    </div>
  );
}

/**
 * Build the query string fragment that the API expects from a filter state.
 * Empty values are omitted so default endpoint behaviour applies.
 */
export function jobsFilterToQuery(filters: JobsFilterState): URLSearchParams {
  const params = new URLSearchParams();
  for (const status of filters.status) {
    params.append("status", status);
  }
  if (filters.submitted_after) {
    params.set("submitted_after", filters.submitted_after);
  }
  if (filters.submitted_before) {
    params.set("submitted_before", filters.submitted_before);
  }
  if (filters.q) {
    params.set("q", filters.q);
  }
  if (filters.sort && filters.sort !== "submitted_at_desc") {
    params.set("sort", filters.sort);
  }
  return params;
}
