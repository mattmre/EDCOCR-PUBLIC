"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/cn";
import type {
  FeatureFlag,
  FeatureFlagCategory,
  FeatureFlagSource,
} from "@/lib/types";

const STORAGE_KEY = "ocr-local:flags-ui";

const CATEGORY_ORDER: FeatureFlagCategory[] = [
  "translation",
  "custody",
  "pipeline",
  "operations",
  "experimental",
];

const CATEGORY_LABEL: Record<string, string> = {
  translation: "Translation",
  custody: "Custody & Forensics",
  pipeline: "Pipeline",
  operations: "Operations",
  experimental: "Experimental",
};

const SOURCE_BADGE_CLASS: Record<FeatureFlagSource, string> = {
  env: "bg-blue-100 text-blue-800 border-blue-200",
  config: "bg-green-100 text-green-800 border-green-200",
  database: "bg-purple-100 text-purple-800 border-purple-200",
  default: "bg-gray-100 text-gray-700 border-gray-200",
};

const SECRET_SUFFIXES = ["_token", "_secret", "_key"];

function isSecretLike(key: string): boolean {
  const lower = key.toLowerCase();
  return SECRET_SUFFIXES.some((suf) => lower.endsWith(suf));
}

function maskValue(raw: string): string {
  if (raw.length <= 4) return "***";
  return `${raw.slice(0, 2)}***${raw.slice(-2)}`;
}

interface ValuePillProps {
  flag: FeatureFlag;
}

function ValuePill({ flag }: ValuePillProps) {
  const v = flag.current_value;
  if (flag.value_type === "boolean") {
    const on = v === true;
    return (
      <span
        data-testid={`flag-value-${flag.key}`}
        className={cn(
          "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
          on
            ? "bg-emerald-100 text-emerald-800 border border-emerald-200"
            : "bg-gray-100 text-gray-600 border border-gray-200"
        )}
      >
        {on ? "ON" : "OFF"}
      </span>
    );
  }
  if (flag.value_type === "enum") {
    return (
      <span
        data-testid={`flag-value-${flag.key}`}
        className="inline-flex items-center rounded border border-border bg-muted px-2 py-0.5 text-xs"
      >
        {String(v ?? "—")}
      </span>
    );
  }
  if (flag.value_type === "integer") {
    return (
      <span
        data-testid={`flag-value-${flag.key}`}
        className="font-mono text-xs"
      >
        {v === null || v === undefined ? "—" : String(v)}
      </span>
    );
  }
  // string
  const raw = v === null || v === undefined ? "—" : String(v);
  const masked = isSecretLike(flag.key) && raw !== "—" ? maskValue(raw) : raw;
  const truncated = masked.length > 24 ? `${masked.slice(0, 24)}…` : masked;
  return (
    <span
      data-testid={`flag-value-${flag.key}`}
      title={raw}
      className="font-mono text-xs text-muted-foreground"
    >
      {truncated}
    </span>
  );
}

function loadCollapsedState(): Record<string, boolean> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as { collapsed?: Record<string, boolean> };
    return parsed.collapsed ?? {};
  } catch {
    return {};
  }
}

function persistCollapsedState(collapsed: Record<string, boolean>): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ collapsed })
    );
  } catch {
    // ignore quota / private mode failures
  }
}

export interface FeatureFlagsListProps {
  flags: FeatureFlag[];
  loading?: boolean;
}

/**
 * Read-only flag table grouped by category. Each row links to the detail
 * page where a change-request can be filed; the list itself never mutates
 * server state.
 */
export function FeatureFlagsList({ flags, loading }: FeatureFlagsListProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setCollapsed(loadCollapsedState());
  }, []);

  const grouped = useMemo(() => {
    const out: Record<string, FeatureFlag[]> = {};
    for (const flag of flags) {
      const cat = String(flag.category);
      if (!out[cat]) out[cat] = [];
      out[cat].push(flag);
    }
    for (const cat of Object.keys(out)) {
      out[cat]!.sort((a, b) => a.key.localeCompare(b.key));
    }
    return out;
  }, [flags]);

  const orderedCategories = useMemo(() => {
    const known = CATEGORY_ORDER.filter((c) => grouped[c]?.length);
    const unknown = Object.keys(grouped)
      .filter((c) => !CATEGORY_ORDER.includes(c as FeatureFlagCategory))
      .sort();
    return [...known, ...unknown];
  }, [grouped]);

  function toggle(category: string) {
    setCollapsed((prev) => {
      const next = { ...prev, [category]: !prev[category] };
      persistCollapsedState(next);
      return next;
    });
  }

  if (loading && flags.length === 0) {
    return (
      <div className="rounded-md border border-border p-6 text-sm text-muted-foreground">
        Loading feature flags…
      </div>
    );
  }

  if (flags.length === 0) {
    return (
      <div
        data-testid="flags-empty"
        className="rounded-md border border-border p-6 text-sm text-muted-foreground"
      >
        No feature flags reported. The flag registry may be empty or the
        backend admin surface may be disabled.
      </div>
    );
  }

  return (
    <div className="space-y-4" data-testid="flags-list">
      {orderedCategories.map((category) => {
        const list = grouped[category] ?? [];
        const isCollapsed = !!collapsed[category];
        return (
          <section
            key={category}
            data-testid={`flag-category-${category}`}
            className="rounded-md border border-border bg-background"
          >
            <button
              type="button"
              onClick={() => toggle(category)}
              data-testid={`flag-category-toggle-${category}`}
              aria-expanded={!isCollapsed}
              aria-controls={`flag-category-body-${category}`}
              className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold hover:bg-muted/30"
            >
              <span>
                {CATEGORY_LABEL[category] ?? category}
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  ({list.length})
                </span>
              </span>
              <span className="text-xs text-muted-foreground">
                {isCollapsed ? "Show" : "Hide"}
              </span>
            </button>
            {!isCollapsed ? (
              <div
                id={`flag-category-body-${category}`}
                className="overflow-x-auto border-t border-border"
              >
                <table className="w-full text-sm" data-testid={`flag-table-${category}`}>
                  <thead className="bg-muted/30 text-xs uppercase text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left">Key</th>
                      <th className="px-3 py-2 text-left">Value</th>
                      <th className="px-3 py-2 text-left">Default</th>
                      <th className="px-3 py-2 text-left">Source</th>
                      <th className="px-3 py-2 text-left">Description</th>
                      <th className="px-3 py-2 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((flag) => (
                      <tr
                        key={flag.key}
                        className="border-t border-border hover:bg-muted/20"
                        data-testid={`flag-row-${flag.key}`}
                      >
                        <td className="px-3 py-2 font-mono text-xs">{flag.key}</td>
                        <td className="px-3 py-2">
                          <ValuePill flag={flag} />
                          {flag.requires_strong_auth ? (
                            <span
                              className="ml-2 inline-flex items-center rounded border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-900"
                              data-testid={`flag-strongauth-${flag.key}`}
                              title="Strong-auth required to change"
                            >
                              SA
                            </span>
                          ) : null}
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                          {String(flag.default_value ?? "—")}
                        </td>
                        <td className="px-3 py-2">
                          <span
                            data-testid={`flag-source-${flag.key}`}
                            className={cn(
                              "inline-flex rounded-full border px-2 py-0.5 text-xs font-medium",
                              SOURCE_BADGE_CLASS[flag.source]
                            )}
                          >
                            {flag.source}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-xs text-muted-foreground">
                          {flag.description || "—"}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <Link
                            href={`/admin/features/${encodeURIComponent(flag.key)}`}
                            className="text-xs text-primary hover:underline"
                            data-testid={`flag-link-${flag.key}`}
                          >
                            View
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
