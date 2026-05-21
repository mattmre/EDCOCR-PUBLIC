"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useRequireAuth } from "@/lib/auth";
import { ApiError } from "@/lib/api-client";
import {
  listTenants,
  upsertTenantConfig,
  getTenantConfig,
  listGlossary,
} from "@/lib/tenant-api";
import type {
  TenantConfig,
  TenantSummary,
} from "@/lib/types";

interface TenantRow {
  tenant_id: string;
  status: string;
  tier: string;
  config: TenantConfig | null;
  glossaryCount: number | null;
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

async function buildRow(tenantId: string, summary?: TenantSummary): Promise<TenantRow> {
  const status = summary?.status ?? "—";
  const tier = summary?.tier ?? "—";
  let config: TenantConfig | null = null;
  try {
    config = await getTenantConfig(tenantId);
  } catch (err) {
    if (!(err instanceof ApiError && err.status === 404)) {
      // Surface non-404s as missing config; the row stays clickable.
      config = null;
    }
  }
  let glossaryCount: number | null = null;
  try {
    const list = await listGlossary(tenantId, { page: 1, page_size: 1 });
    glossaryCount = list.total;
  } catch {
    glossaryCount = null;
  }
  return { tenant_id: tenantId, status, tier, config, glossaryCount };
}

export default function TenantsListPage() {
  useRequireAuth();
  const [rows, setRows] = useState<TenantRow[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [manualMode, setManualMode] = useState<boolean>(false);
  const [manualInput, setManualInput] = useState<string>("");
  const [showCreate, setShowCreate] = useState<boolean>(false);
  const [createDraft, setCreateDraft] = useState<{
    tenant_id: string;
    target_languages: string;
  }>({ tenant_id: "", target_languages: "en" });
  const [createBusy, setCreateBusy] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listTenants()
      .then(async (tenants) => {
        if (cancelled) return;
        const enriched = await Promise.all(
          tenants.map((t) => buildRow(t.tenant_id, t))
        );
        if (cancelled) return;
        setRows(enriched);
      })
      .catch((err) => {
        if (cancelled) return;
        // 401/403/404 from /api/v1/admin/tenants -- fall back to manual mode.
        if (err instanceof ApiError && [403, 404, 501].includes(err.status)) {
          setManualMode(true);
        } else {
          setError(describeError(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return => {
      cancelled = true;
    };
  }, []);

  async function handleManualLoad() {
    const tenantId = manualInput.trim();
    if (!tenantId) return;
    setLoading(true);
    setError(null);
    try {
      const row = await buildRow(tenantId);
      setRows((existing) => {
        const filtered = existing.filter((r) => r.tenant_id !== tenantId);
        return [row, ...filtered];
      });
    } catch (err) {
      setError(describeError(err));
    } finally {
      setLoading(false);
      setManualInput("");
    }
  }

  async function handleCreateSubmit() {
    const tenantId = createDraft.tenant_id.trim();
    if (!tenantId) {
      setError("tenant_id is required");
      return;
    }
    setCreateBusy(true);
    setError(null);
    try {
      const target = createDraft.target_languages
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      await upsertTenantConfig(tenantId, {
        target_languages: target,
        preferred_engines: [],
        allow_nc_licensed: false,
        require_certified: false,
        default_quality_tier: "standard",
      });
      const row = await buildRow(tenantId);
      setRows((existing) => {
        const filtered = existing.filter((r) => r.tenant_id !== tenantId);
        return [row, ...filtered];
      });
      setShowCreate(false);
      setCreateDraft({ tenant_id: "", target_languages: "en" });
    } catch (err) {
      setError(describeError(err));
    } finally {
      setCreateBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Tenants</h1>
          <p className="text-sm text-muted-foreground">
            Translation engine policy and per-tenant glossary entries. Engine and licensing
            settings are read at translation routing time; glossary entries are applied during
            assembly.
          </p>
        </div>
        <Button
          type="button"
          onClick={() => setShowCreate(true)}
          data-testid="tenants-create-button"
        >
          Create new tenant
        </Button>
      </div>

      {error ? (
        <p className="text-sm text-destructive" role="alert" data-testid="tenants-error">
          {error}
        </p>
      ) : null}

      {showCreate ? (
        <Card data-testid="tenants-create-modal">
          <CardHeader>
            <CardTitle>New tenant config</CardTitle>
            <CardDescription>
              Creates or replaces the translation config row for the supplied tenant_id.
              No backend tenant directory entry is provisioned.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              placeholder="tenant_id"
              value={createDraft.tenant_id}
              onChange={(e) =>
                setCreateDraft((d) => ({ ...d, tenant_id: e.target.value }))
              }
              data-testid="tenants-create-id"
            />
            <Input
              placeholder="target languages (comma separated, e.g. en,fr,de)"
              value={createDraft.target_languages}
              onChange={(e) =>
                setCreateDraft((d) => ({ ...d, target_languages: e.target.value }))
              }
              data-testid="tenants-create-langs"
            />
            <div className="flex gap-2">
              <Button
                type="button"
                onClick={handleCreateSubmit}
                disabled={createBusy}
                data-testid="tenants-create-submit"
              >
                {createBusy ? "Saving…" : "Create"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setShowCreate(false);
                  setCreateDraft({ tenant_id: "", target_languages: "en" });
                }}
              >
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {manualMode ? (
        <Card data-testid="tenants-manual-mode">
          <CardHeader>
            <CardTitle>Open a tenant by id</CardTitle>
            <CardDescription>
              The admin tenant directory is unavailable — likely because multi-tenancy
              is disabled or this key lacks platform-admin scope. Enter a tenant_id to
              load its translation config directly.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Input
                placeholder="tenant_id"
                value={manualInput}
                onChange={(e) => setManualInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void handleManualLoad();
                  }
                }}
                data-testid="tenants-manual-input"
              />
              <Button
                type="button"
                onClick={handleManualLoad}
                data-testid="tenants-manual-load"
              >
                Open
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : null}

      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm" data-testid="tenants-table">
          <thead className="bg-muted/30 text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left">Tenant</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Tier</th>
              <th className="px-3 py-2 text-left">Engines</th>
              <th className="px-3 py-2 text-left">NC licensed</th>
              <th className="px-3 py-2 text-left">Target langs</th>
              <th className="px-3 py-2 text-left">Quality tier</th>
              <th className="px-3 py-2 text-right">Glossary</th>
            </tr>
          </thead>
          <tbody>
            {loading && rows.length === 0 ? (
              <tr>
                <td className="px-3 py-2 text-muted-foreground" colSpan={8}>
                  Loading…
                </td>
              </tr>
            ) : rows.length === 0 ? (
              <tr>
                <td
                  className="px-3 py-2 text-muted-foreground"
                  colSpan={8}
                  data-testid="tenants-empty"
                >
                  No tenants yet.
                </td>
              </tr>
            ) : (
              rows.map((row) => (
                <tr
                  key={row.tenant_id}
                  className="border-t border-border hover:bg-muted/20"
                  data-testid={`tenant-row-${row.tenant_id}`}
                >
                  <td className="px-3 py-2 font-mono">
                    <Link
                      href={`/admin/tenants/${encodeURIComponent(row.tenant_id)}`}
                      className="text-primary hover:underline"
                      data-testid={`tenant-link-${row.tenant_id}`}
                    >
                      {row.tenant_id}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-xs">{row.status}</td>
                  <td className="px-3 py-2 text-xs">{row.tier}</td>
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {row.config?.preferred_engines && row.config.preferred_engines.length > 0 ? (
                        row.config.preferred_engines.map((eng) => (
                          <span
                            key={eng}
                            className="rounded-full border border-border bg-muted px-2 py-0.5 text-xs"
                          >
                            {eng}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {row.config ? (
                      row.config.allow_nc_licensed ? (
                        <span
                          className="rounded-full border border-amber-500/50 bg-amber-500/10 px-2 py-0.5 text-amber-700"
                          data-testid={`tenant-nc-${row.tenant_id}`}
                        >
                          NC allowed
                        </span>
                      ) : (
                        <span className="text-muted-foreground">commercial-safe</span>
                      )
                    ) : (
                      <span className="text-muted-foreground">no config</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {row.config?.target_languages.join(", ") || "—"}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {row.config?.default_quality_tier ?? "—"}
                  </td>
                  <td className="px-3 py-2 text-right text-xs">
                    {row.glossaryCount === null ? "—" : row.glossaryCount}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
