"use client";

import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  createGlossaryEntry,
  deleteGlossaryEntry,
  listGlossary,
  updateGlossaryEntry,
  type GlossaryEntryInput,
} from "@/lib/tenant-api";
import { parseCsv, serializeCsv, triggerDownload } from "@/lib/csv";
import { cn } from "@/lib/cn";
import { ApiError } from "@/lib/api-client";
import type { GlossaryEntry, GlossaryFilters } from "@/lib/types";

const DEFAULT_PAGE_SIZE = 100;

const CSV_COLUMNS = [
  "source_term",
  "target_term",
  "source_lang",
  "target_lang",
  "case_sensitive",
  "is_regex",
  "priority",
  "notes",
];

export interface GlossaryEditorProps {
  tenantId: string;
}

interface BlankEntry extends GlossaryEntryInput {
  case_sensitive: boolean;
  is_regex: boolean;
  priority: number;
  notes: string;
}

function blankEntry(): BlankEntry {
  return {
    source_term: "",
    target_term: "",
    source_lang: "en",
    target_lang: "es",
    case_sensitive: false,
    is_regex: false,
    priority: 100,
    notes: "",
  };
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

/**
 * Coerce truthy/falsey CSV cell values into a boolean. Accepts "true",
 * "false", "1", "0", "yes", "no" (case-insensitive). Empty -> false.
 */
function csvBool(raw: string | undefined): boolean {
  if (!raw) return false;
  return ["1", "true", "yes", "y"].includes(raw.trim().toLowerCase());
}

function csvInt(raw: string | undefined, fallback: number): number {
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

export function GlossaryEditor({ tenantId }: GlossaryEditorProps) {
  const [entries, setEntries] = useState<GlossaryEntry[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<GlossaryFilters>({});
  const [page, setPage] = useState<number>(1);
  const [showCreate, setShowCreate] = useState<boolean>(false);
  const [createDraft, setCreateDraft] = useState<BlankEntry>(blankEntry());
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState<Partial<GlossaryEntryInput>>({});
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const load = useMemo( => async => {
      setLoading(true);
      try {
        const res = await listGlossary(tenantId, {
          ...filters,
          page,
          page_size: DEFAULT_PAGE_SIZE,
        });
        setEntries(res.entries);
        setTotal(res.total);
        setError(null);
      } catch (err) {
        setError(describeError(err));
      } finally {
        setLoading(false);
      }
    },
    [tenantId, filters, page]
  );

  useEffect(() => {
    void load();
  }, [load]);

  async function handleCreate() {
    if (!createDraft.source_term.trim() || !createDraft.target_term.trim()) {
      setError("Source and target terms are required");
      return;
    }
    try {
      await createGlossaryEntry(tenantId, createDraft);
      setShowCreate(false);
      setCreateDraft(blankEntry());
      await load();
    } catch (err) {
      setError(describeError(err));
    }
  }

  async function handleSaveEdit(id: number) {
    const baseline = entries.find((e) => e.id === id);
    if (!baseline) return;
    // Optimistic update.
    const optimistic: GlossaryEntry = {
      ...baseline,
      ...(editDraft as Partial<GlossaryEntry>),
    };
    setEntries((rows) => rows.map((r) => (r.id === id ? optimistic : r)));
    setEditingId(null);
    try {
      const updated = await updateGlossaryEntry(tenantId, id, editDraft);
      setEntries((rows) => rows.map((r) => (r.id === id ? updated : r)));
    } catch (err) {
      setError(describeError(err));
      // Roll back.
      setEntries((rows) => rows.map((r) => (r.id === id ? baseline : r)));
    }
  }

  async function handleDelete(id: number) {
    const baseline = entries.find((e) => e.id === id);
    if (!baseline) return;
    setEntries((rows) => rows.filter((r) => r.id !== id));
    try {
      await deleteGlossaryEntry(tenantId, id);
    } catch (err) {
      setError(describeError(err));
      setEntries((rows) => [...rows, baseline]);
    }
  }

  function handleExport() {
    const records = entries.map((e) => ({
      source_term: e.source_term,
      target_term: e.target_term,
      source_lang: e.source_lang,
      target_lang: e.target_lang,
      case_sensitive: String(e.case_sensitive),
      is_regex: String(e.is_regex),
      priority: String(e.priority),
      notes: e.notes ?? "",
    }));
    const csv = serializeCsv(records, CSV_COLUMNS, { bom: true });
    triggerDownload(`glossary-${tenantId}.csv`, csv);
  }

  async function readFileText(file: File): Promise<string> {
    if (typeof file.text === "function") {
      return file.text();
    }
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = => reject(reader.error ?? new Error("FileReader error"));
      reader.onload = => resolve(String(reader.result ?? ""));
      reader.readAsText(file);
    });
  }

  async function handleImportFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setImportStatus("Parsing…");
    try {
      const text = await readFileText(file);
      const records = parseCsv(text);
      let created = 0;
      let skipped = 0;
      for (const row of records) {
        const payload: GlossaryEntryInput = {
          source_term: (row.source_term ?? "").trim(),
          target_term: (row.target_term ?? "").trim(),
          source_lang: (row.source_lang ?? "").trim(),
          target_lang: (row.target_lang ?? "").trim(),
          case_sensitive: csvBool(row.case_sensitive),
          is_regex: csvBool(row.is_regex),
          priority: csvInt(row.priority, 100),
          notes: row.notes ?? "",
        };
        if (
          !payload.source_term ||
          !payload.target_term ||
          !payload.source_lang ||
          !payload.target_lang
        ) {
          skipped += 1;
          continue;
        }
        try {
          await createGlossaryEntry(tenantId, payload);
          created += 1;
        } catch {
          skipped += 1;
        }
      }
      setImportStatus(`Imported ${created} entry/entries (${skipped} skipped)`);
      await load();
    } catch (err) {
      setImportStatus(`Import failed: ${describeError(err)}`);
    } finally {
      event.target.value = "";
    }
  }

  function startEdit(row: GlossaryEntry) {
    setEditingId(row.id);
    setEditDraft({
      source_term: row.source_term,
      target_term: row.target_term,
      source_lang: row.source_lang,
      target_lang: row.target_lang,
      case_sensitive: row.case_sensitive,
      is_regex: row.is_regex,
      priority: row.priority,
      notes: row.notes ?? "",
    });
  }

  return (
    <div className="space-y-4" data-testid="glossary-editor">
      {error ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-sm text-destructive"
          data-testid="glossary-error"
        >
          {error}
        </div>
      ) : null}

      <div className="flex flex-wrap items-end gap-2">
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">Source language</span>
          <Input
            value={filters.source_lang ?? ""}
            onChange={(e) => {
              setPage(1);
              setFilters((f) => ({ ...f, source_lang: e.target.value || undefined }));
            }}
            placeholder="any"
            className="w-28"
            data-testid="glossary-filter-source-lang"
          />
        </label>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">Target language</span>
          <Input
            value={filters.target_lang ?? ""}
            onChange={(e) => {
              setPage(1);
              setFilters((f) => ({ ...f, target_lang: e.target.value || undefined }));
            }}
            placeholder="any"
            className="w-28"
            data-testid="glossary-filter-target-lang"
          />
        </label>
        <div className="ml-auto flex gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            data-testid="glossary-import-button"
          >
            Import CSV
          </Button>
          <input
            type="file"
            accept="text/csv,.csv"
            className="hidden"
            ref={fileInputRef}
            onChange={handleImportFile}
            data-testid="glossary-import-input"
          />
          <Button
            type="button"
            variant="outline"
            onClick={handleExport}
            data-testid="glossary-export-button"
          >
            Export CSV
          </Button>
          <Button
            type="button"
            onClick={() => setShowCreate(true)}
            data-testid="glossary-add-button"
          >
            Add entry
          </Button>
        </div>
      </div>

      {importStatus ? (
        <p className="text-xs text-muted-foreground" data-testid="glossary-import-status">
          {importStatus}
        </p>
      ) : null}

      {showCreate ? (
        <div
          className="space-y-2 rounded-md border border-border bg-muted/30 p-3"
          data-testid="glossary-create-modal"
        >
          <div className="grid grid-cols-2 gap-2">
            <Input
              placeholder="source term"
              value={createDraft.source_term}
              onChange={(e) =>
                setCreateDraft((d) => ({ ...d, source_term: e.target.value }))
              }
              data-testid="glossary-create-source-term"
            />
            <Input
              placeholder="target term"
              value={createDraft.target_term}
              onChange={(e) =>
                setCreateDraft((d) => ({ ...d, target_term: e.target.value }))
              }
              data-testid="glossary-create-target-term"
            />
            <Input
              placeholder="source lang"
              value={createDraft.source_lang}
              onChange={(e) =>
                setCreateDraft((d) => ({ ...d, source_lang: e.target.value }))
              }
              data-testid="glossary-create-source-lang"
            />
            <Input
              placeholder="target lang"
              value={createDraft.target_lang}
              onChange={(e) =>
                setCreateDraft((d) => ({ ...d, target_lang: e.target.value }))
              }
              data-testid="glossary-create-target-lang"
            />
          </div>
          <div className="flex items-center gap-3 text-xs">
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={createDraft.case_sensitive}
                onChange={(e) =>
                  setCreateDraft((d) => ({ ...d, case_sensitive: e.target.checked }))
                }
              />
              Case-sensitive
            </label>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={createDraft.is_regex}
                onChange={(e) =>
                  setCreateDraft((d) => ({ ...d, is_regex: e.target.checked }))
                }
              />
              Regex
            </label>
            <label className="flex items-center gap-1">
              <span>Priority</span>
              <Input
                type="number"
                value={String(createDraft.priority)}
                onChange={(e) =>
                  setCreateDraft((d) => ({
                    ...d,
                    priority: Number.parseInt(e.target.value, 10) || 0,
                  }))
                }
                className="w-20"
              />
            </label>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              onClick={handleCreate}
              data-testid="glossary-create-submit"
            >
              Create
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setShowCreate(false);
                setCreateDraft(blankEntry());
              }}
            >
              Cancel
            </Button>
          </div>
        </div>
      ) : null}

      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm" data-testid="glossary-table">
          <thead className="bg-muted/30 text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left">Source</th>
              <th className="px-3 py-2 text-left">Target</th>
              <th className="px-3 py-2 text-left">Languages</th>
              <th className="px-3 py-2 text-left">Flags</th>
              <th className="px-3 py-2 text-left">Priority</th>
              <th className="px-3 py-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td className="px-3 py-2 text-muted-foreground" colSpan={6}>
                  Loading…
                </td>
              </tr>
            ) : entries.length === 0 ? (
              <tr>
                <td
                  className="px-3 py-2 text-muted-foreground"
                  colSpan={6}
                  data-testid="glossary-empty"
                >
                  No glossary entries.
                </td>
              </tr>
            ) : (
              entries.map((row) => {
                const editing = editingId === row.id;
                return (
                  <tr
                    key={row.id}
                    className="border-t border-border"
                    data-testid={`glossary-row-${row.id}`}
                  >
                    <td className={cn("px-3 py-1", editing ? "" : "font-medium")}>
                      {editing ? (
                        <Input
                          value={(editDraft.source_term ?? row.source_term) as string}
                          onChange={(e) =>
                            setEditDraft((d) => ({ ...d, source_term: e.target.value }))
                          }
                          data-testid={`glossary-edit-source-term-${row.id}`}
                        />
                      ) : (
                        row.source_term
                      )}
                    </td>
                    <td className="px-3 py-1">
                      {editing ? (
                        <Input
                          value={(editDraft.target_term ?? row.target_term) as string}
                          onChange={(e) =>
                            setEditDraft((d) => ({ ...d, target_term: e.target.value }))
                          }
                          data-testid={`glossary-edit-target-term-${row.id}`}
                        />
                      ) : (
                        row.target_term
                      )}
                    </td>
                    <td className="px-3 py-1 text-xs text-muted-foreground">
                      {row.source_lang} → {row.target_lang}
                    </td>
                    <td className="px-3 py-1 text-xs text-muted-foreground">
                      {row.case_sensitive ? "case " : ""}
                      {row.is_regex ? "regex " : ""}
                    </td>
                    <td className="px-3 py-1 text-xs">{row.priority}</td>
                    <td className="px-3 py-1 text-right">
                      {editing ? (
                        <div className="flex justify-end gap-1">
                          <Button
                            type="button"
                            size="sm"
                            onClick={() => handleSaveEdit(row.id)}
                            data-testid={`glossary-save-${row.id}`}
                          >
                            Save
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              setEditingId(null);
                              setEditDraft({});
                            }}
                          >
                            Cancel
                          </Button>
                        </div>
                      ) : (
                        <div className="flex justify-end gap-1">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            onClick={() => startEdit(row)}
                            data-testid={`glossary-edit-${row.id}`}
                          >
                            Edit
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="destructive"
                            onClick={() => handleDelete(row.id)}
                            data-testid={`glossary-delete-${row.id}`}
                          >
                            Delete
                          </Button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-muted-foreground" data-testid="glossary-summary">
        Showing {entries.length} of {total} entries
      </p>
    </div>
  );
}
