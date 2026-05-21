"use client";

import { useState } from "react";
import { cn } from "@/lib/cn";
import type { CustodyEvent } from "@/lib/audit-verify";
import { JsonTree } from "./json-tree";

interface AuditEventCardProps {
  event: CustodyEvent;
  index: number;
  /** True when the chain verifier marked this event as the first broken link. */
  broken?: boolean;
  /** True when the verifier confirmed this event as intact. */
  verified?: boolean;
}

const EVENT_TYPE_COLORS: Record<string, string> = {
  file_ingested: "bg-sky-100 text-sky-900",
  page_extracted: "bg-sky-50 text-sky-900",
  ocr_primary: "bg-emerald-50 text-emerald-900",
  ocr_fallback: "bg-amber-50 text-amber-900",
  ocr_image_only: "bg-orange-50 text-orange-900",
  language_detected: "bg-indigo-50 text-indigo-900",
  language_reprocess: "bg-indigo-50 text-indigo-900",
  docintel_analysis: "bg-violet-50 text-violet-900",
  assembly_complete: "bg-emerald-50 text-emerald-900",
  compression_complete: "bg-emerald-50 text-emerald-900",
  dpi_escalation: "bg-amber-50 text-amber-900",
  processing_failed: "bg-red-50 text-red-900",
  TRANSLATION_APPLIED: "bg-teal-50 text-teal-900",
  TRANSLATION_REJECTED: "bg-red-50 text-red-900",
  LANGUAGE_DETECTED: "bg-indigo-50 text-indigo-900",
  LANGUAGE_MIXED_SCRIPT: "bg-indigo-50 text-indigo-900",
  LANGUAGE_REDETECTED: "bg-indigo-50 text-indigo-900",
};

function eventTypeBadgeClass(type: string): string {
  return EVENT_TYPE_COLORS[type] ?? "bg-muted text-foreground";
}

function shortHash(h: string | null | undefined): string {
  if (!h) return "—";
  return h.length > 12 ? `${h.slice(0, 8)}…${h.slice(-4)}` : h;
}

export function AuditEventCard({ event, index, broken, verified }: AuditEventCardProps) {
  const [open, setOpen] = useState(false);
  const [hashOpen, setHashOpen] = useState(false);

  const actor =
    typeof event.data === "object" && event.data !== null
      ? (event.data as Record<string, unknown>).actor ??
        (event.data as Record<string, unknown>).user ??
        "system"
      : "system";

  return (
    <li
      className={cn(
        "relative border-l-2 pl-6",
        broken
          ? "border-red-500"
          : verified
          ? "border-emerald-500"
          : "border-border"
      )}
      data-testid="audit-event-card"
      data-event-type={event.event_type}
      data-event-index={index}
    >
      <div className="absolute -left-[7px] top-1 h-3 w-3 rounded-full border bg-background"
        style={{
          borderColor: broken
            ? "rgb(239 68 68)"
            : verified
            ? "rgb(16 185 129)"
            : undefined,
        }}
      />
      <div className="rounded-md border border-border bg-background p-3 shadow-sm">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground tabular-nums">
            #{index}
          </span>
          <span
            className={cn(
              "rounded px-2 py-0.5 text-xs font-medium",
              eventTypeBadgeClass(event.event_type)
            )}
          >
            {event.event_type}
          </span>
          <span className="text-xs text-muted-foreground">
            {event.timestamp}
          </span>
          <span className="text-xs text-muted-foreground">
            actor: <span className="font-medium text-foreground">{String(actor)}</span>
          </span>
          {broken ? (
            <span className="rounded bg-red-100 px-2 py-0.5 text-xs font-semibold text-red-900">
              BROKEN
            </span>
          ) : null}
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="ml-auto text-xs font-medium text-primary hover:underline"
            aria-expanded={open}
          >
            {open ? "Hide payload" : "Show payload"}
          </button>
        </div>

        <div className="mt-2 grid grid-cols-1 gap-1 text-xs sm:grid-cols-2">
          <button
            type="button"
            className="text-left font-mono text-xs text-muted-foreground hover:text-foreground"
            onClick={() => setHashOpen((v) => !v)}
            aria-expanded={hashOpen}
            title="Click to expand full hash"
          >
            <span className="font-semibold not-italic text-foreground">hash</span>{" "}
            {hashOpen ? event.hash : shortHash(event.hash)}
          </button>
          <span className="font-mono text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">prev</span>{" "}
            {hashOpen ? (event.prev_hash ?? "null") : shortHash(event.prev_hash)}
          </span>
        </div>

        {open ? (
          <div className="mt-3 rounded border border-border bg-muted/40 p-2">
            <JsonTree value={event.data} />
          </div>
        ) : null}
      </div>
    </li>
  );
}
