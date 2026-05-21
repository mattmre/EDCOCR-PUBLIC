"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { ReviewQueueFilters, ReviewStatus } from "@/lib/types";

const STATUS_OPTIONS: Array<{ value: ReviewStatus; label: string }> = [
  { value: "pending", label: "Pending" },
  { value: "approved", label: "Approved" },
  { value: "rejected", label: "Rejected" },
  { value: "reprocess", label: "Reprocess" },
];

const REASON_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "", label: "All reasons" },
  { value: "low_confidence", label: "Low confidence" },
  { value: "degraded_quality", label: "Degraded quality" },
  { value: "handwriting_detected", label: "Handwriting detected" },
  { value: "dpi_escalation_failed", label: "DPI escalation failed" },
  { value: "image_only_pages", label: "Image-only pages" },
  { value: "classification_uncertain", label: "Classification uncertain" },
  { value: "manual_flag", label: "Manual flag" },
];

const SEARCH_DEBOUNCE_MS = 300;

export const EMPTY_REVIEW_FILTER_STATE: ReviewQueueFilters = {
  status: ["pending"],
  reason: "",
  q: "",
};

export interface ReviewQueueFiltersProps {
  value: ReviewQueueFilters;
  onChange: (next: ReviewQueueFilters) => void;
  /** Debounce window for the free-text search box, in milliseconds. */
  searchDebounceMs?: number;
}

/**
 * D8 review queue filter bar.
 *
 * - Status is a multi-select chip group (default: pending).
 * - Reason is a single-select dropdown matching backend ReviewReason enum.
 * - Search has a 300 ms debounce so we don't fire one fetch per keystroke.
 */
export function ReviewQueueFiltersBar({
  value,
  onChange,
  searchDebounceMs = SEARCH_DEBOUNCE_MS,
}: ReviewQueueFiltersProps) {
  const [searchDraft, setSearchDraft] = useState<string>(value.q);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  function toggleStatus(s: ReviewStatus) {
    const isOn = value.status.includes(s);
    const next = isOn
      ? value.status.filter((existing) => existing !== s)
      : [...value.status, s];
    onChange({ ...value, status: next });
  }

  function clearAll() {
    setSearchDraft("");
    onChange(EMPTY_REVIEW_FILTER_STATE);
  }

  const hasAnyFilter =
    value.status.length !== 1 ||
    value.status[0] !== "pending" ||
    value.reason !== "" ||
    value.q !== "";

  return (
    <div
      className="space-y-3 rounded-md border border-border bg-background p-3"
      data-testid="review-filter-bar"
    >
      <div
        className="flex flex-wrap gap-2"
        role="group"
        aria-label="Review status filter"
      >
        {STATUS_OPTIONS.map((opt) => {
          const active = value.status.includes(opt.value);
          return (
            <button
              key={opt.value}
              type="button"
              data-testid={`review-chip-${opt.value}`}
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

      <div className="grid gap-3 md:grid-cols-3">
        <label className="block text-xs font-medium text-muted-foreground">
          Search
          <Input
            data-testid="review-filter-search"
            type="search"
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            placeholder="Job ID or review ID"
            className="mt-1"
            maxLength={256}
            aria-label="Search by job or review id"
          />
        </label>
        <label className="block text-xs font-medium text-muted-foreground">
          Reason
          <select
            data-testid="review-filter-reason"
            value={value.reason}
            onChange={(e) => onChange({ ...value, reason: e.target.value })}
            className="mt-1 block h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
            aria-label="Filter by review reason"
          >
            {REASON_OPTIONS.map((opt) => (
              <option key={opt.value || "_all"} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <div className="flex items-end justify-end">
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAll}
            disabled={!hasAnyFilter}
            data-testid="review-filter-clear"
          >
            Clear filters
          </Button>
        </div>
      </div>
    </div>
  );
}
