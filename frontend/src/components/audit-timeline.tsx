"use client";

import { useMemo, useState } from "react";
import type { CustodyEvent, VerificationResult } from "@/lib/audit-verify";
import { AuditEventCard } from "./audit-event-card";

interface AuditTimelineProps {
  events: CustodyEvent[];
  verification: VerificationResult | null;
}

interface FilterState {
  selectedTypes: Set<string>;
  actor: string;
  startTimestamp: string;
  endTimestamp: string;
}

function getActor(event: CustodyEvent): string {
  if (event.data && typeof event.data === "object") {
    const data = event.data as Record<string, unknown>;
    const a = data.actor ?? data.user;
    if (typeof a === "string") {
      return a;
    }
  }
  return "system";
}

export function AuditTimeline({ events, verification }: AuditTimelineProps) {
  const allTypes = useMemo(() => {
    const set = new Set<string>();
    events.forEach((e) => set.add(e.event_type));
    return Array.from(set).sort();
  }, [events]);

  const allActors = useMemo(() => {
    const set = new Set<string>();
    events.forEach((e) => set.add(getActor(e)));
    return Array.from(set).sort();
  }, [events]);

  const timestamps = useMemo(() => {
    const sorted = events.map((e) => e.timestamp).sort();
    return {
      first: sorted[0] ?? "",
      last: sorted[sorted.length - 1] ?? "",
    };
  }, [events]);

  const [filters, setFilters] = useState<FilterState>({
    selectedTypes: new Set<string>(),
    actor: "",
    startTimestamp: "",
    endTimestamp: "",
  });

  const filteredEvents = useMemo(() => {
    return events.filter((e) => {
      if (filters.selectedTypes.size > 0 && !filters.selectedTypes.has(e.event_type)) {
        return false;
      }
      if (filters.actor && getActor(e) !== filters.actor) {
        return false;
      }
      if (filters.startTimestamp && e.timestamp < filters.startTimestamp) {
        return false;
      }
      if (filters.endTimestamp && e.timestamp > filters.endTimestamp) {
        return false;
      }
      return true;
    });
  }, [events, filters]);

  function toggleType(type: string) {
    setFilters((prev) => {
      const next = new Set(prev.selectedTypes);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return { ...prev, selectedTypes: next };
    });
  }

  function clearFilters() {
    setFilters({
      selectedTypes: new Set<string>(),
      actor: "",
      startTimestamp: "",
      endTimestamp: "",
    });
  }

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-border bg-muted/30 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-muted-foreground">Filters:</span>
          {allTypes.map((t) => {
            const active = filters.selectedTypes.has(t);
            return (
              <button
                key={t}
                type="button"
                onClick={() => toggleType(t)}
                className={
                  active
                    ? "rounded border border-primary bg-primary px-2 py-0.5 text-xs text-primary-foreground"
                    : "rounded border border-border bg-background px-2 py-0.5 text-xs"
                }
                aria-pressed={active}
              >
                {t}
              </button>
            );
          })}
          {filters.selectedTypes.size > 0 ||
          filters.actor ||
          filters.startTimestamp ||
          filters.endTimestamp ? (
            <button
              type="button"
              onClick={clearFilters}
              className="ml-2 text-xs text-primary hover:underline"
            >
              Clear filters
            </button>
          ) : null}
        </div>

        <div className="mt-3 grid grid-cols-1 gap-2 text-xs sm:grid-cols-3">
          <label className="flex flex-col gap-1">
            <span className="font-semibold text-muted-foreground">Actor</span>
            <select
              value={filters.actor}
              onChange={(e) => setFilters((prev) => ({ ...prev, actor: e.target.value }))}
              className="rounded border border-border bg-background px-2 py-1"
              aria-label="Filter by actor"
            >
              <option value="">All actors</option>
              {allActors.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-semibold text-muted-foreground">From</span>
            <input
              type="text"
              value={filters.startTimestamp}
              onChange={(e) =>
                setFilters((prev) => ({ ...prev, startTimestamp: e.target.value }))
              }
              placeholder={timestamps.first}
              className="rounded border border-border bg-background px-2 py-1 font-mono"
              aria-label="Filter by start timestamp"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-semibold text-muted-foreground">To</span>
            <input
              type="text"
              value={filters.endTimestamp}
              onChange={(e) =>
                setFilters((prev) => ({ ...prev, endTimestamp: e.target.value }))
              }
              placeholder={timestamps.last}
              className="rounded border border-border bg-background px-2 py-1 font-mono"
              aria-label="Filter by end timestamp"
            />
          </label>
        </div>

        <p className="mt-2 text-xs text-muted-foreground">
          Showing {filteredEvents.length} of {events.length} event(s).
        </p>
      </div>

      <ol className="space-y-2" data-testid="audit-timeline">
        {filteredEvents.map((event) => {
          const realIndex = events.indexOf(event);
          const broken =
            verification?.status === "broken" &&
            typeof verification.brokenAtIndex === "number" &&
            verification.brokenAtIndex === realIndex;
          const verified =
            verification?.status === "intact" ||
            (verification?.status === "broken" &&
              typeof verification.brokenAtIndex === "number" &&
              realIndex < verification.brokenAtIndex);
          return (
            <AuditEventCard
              key={event.hash || realIndex}
              event={event}
              index={realIndex}
              broken={broken}
              verified={verified}
            />
          );
        })}
        {filteredEvents.length === 0 ? (
          <li className="rounded border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
            No events match the current filters.
          </li>
        ) : null}
      </ol>
    </div>
  );
}
