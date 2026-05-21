import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

import { GlossaryEditor } from "@/components/GlossaryEditor";
import { setApiKey } from "@/lib/auth";
import { parseCsv } from "@/lib/csv";
import type { GlossaryEntry } from "@/lib/types";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function noContent(): Response {
  return new Response(null, { status: 204 });
}

function entry(partial: Partial<GlossaryEntry> = {}): GlossaryEntry {
  return {
    id: partial.id ?? 1,
    tenant_id: partial.tenant_id ?? "acme",
    source_term: partial.source_term ?? "hello",
    target_term: partial.target_term ?? "hola",
    source_lang: partial.source_lang ?? "en",
    target_lang: partial.target_lang ?? "es",
    case_sensitive: partial.case_sensitive ?? false,
    is_regex: partial.is_regex ?? false,
    priority: partial.priority ?? 100,
    notes: partial.notes ?? "",
    created_at: null,
    updated_at: null,
  };
}

describe("<GlossaryEditor />", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders existing entries from the API", async => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.includes("/translation/tenants/acme/glossary")) {
        return Promise.resolve(
          jsonResponse({
            entries: [entry({ id: 1 }), entry({ id: 2, source_term: "world" })],
            total: 2,
            page: 1,
            page_size: 100,
          })
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<GlossaryEditor tenantId="acme" />);

    await waitFor(() => {
      expect(screen.getByTestId("glossary-row-1")).toBeInTheDocument();
      expect(screen.getByTestId("glossary-row-2")).toBeInTheDocument();
    });
    expect(screen.getByTestId("glossary-summary")).toHaveTextContent("2 of 2");
  });

  it("optimistically updates the row on inline edit and persists via PATCH", async => {
    let patchBody: unknown = null;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.includes("/translation/tenants/acme/glossary") && method === "GET") {
        return Promise.resolve(
          jsonResponse({
            entries: [entry({ id: 7, target_term: "hola" })],
            total: 1,
            page: 1,
            page_size: 100,
          })
        );
      }
      if (
        url.includes("/translation/tenants/acme/glossary/7") &&
        method === "PATCH"
      ) {
        patchBody = init?.body ? JSON.parse(String(init.body)) : null;
        return Promise.resolve(
          jsonResponse(entry({ id: 7, target_term: "hola amigo" }))
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<GlossaryEditor tenantId="acme" />);
    await waitFor(() => {
      expect(screen.getByTestId("glossary-row-7")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("glossary-edit-7"));
    fireEvent.change(screen.getByTestId("glossary-edit-target-term-7"), {
      target: { value: "hola amigo" },
    });
    fireEvent.click(screen.getByTestId("glossary-save-7"));

    await waitFor(() => {
      expect(patchBody).toMatchObject({ target_term: "hola amigo" });
    });
    // The visible row reflects the new value.
    await waitFor(() => {
      const row = screen.getByTestId("glossary-row-7");
      expect(row).toHaveTextContent("hola amigo");
    });
  });

  it("removes a row on delete and rolls back on failure", async => {
    let deleteAttempted = false;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.includes("/glossary") && method === "GET") {
        return Promise.resolve(
          jsonResponse({
            entries: [entry({ id: 9 })],
            total: 1,
            page: 1,
            page_size: 100,
          })
        );
      }
      if (url.includes("/glossary/9") && method === "DELETE") {
        deleteAttempted = true;
        // Server rejects.
        return Promise.resolve(
          jsonResponse({ detail: "permission denied" }, 500)
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<GlossaryEditor tenantId="acme" />);
    await waitFor(() => {
      expect(screen.getByTestId("glossary-row-9")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("glossary-delete-9"));

    await waitFor(() => {
      expect(deleteAttempted).toBe(true);
    });
    // Row should be re-added after server rejects.
    await waitFor(() => {
      expect(screen.getByTestId("glossary-row-9")).toBeInTheDocument();
    });
    expect(screen.getByTestId("glossary-error")).toBeInTheDocument();
  });

  it("creates entries from imported CSV (parsing happy-path)", async => {
    const created: unknown[] = [];
    let listCallCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.includes("/glossary") && !url.match(/\/glossary\/\d+/) && method === "GET") {
        listCallCount += 1;
        const e =
          listCallCount === 1
            ? []
            : [
                entry({ id: 1, source_term: "Hello, World", target_term: "Hola, Mundo" }),
                entry({ id: 2, source_term: 'He said "ok"', target_term: "Dijo ok" }),
              ];
        return Promise.resolve(
          jsonResponse({ entries: e, total: e.length, page: 1, page_size: 100 })
        );
      }
      if (url.includes("/glossary") && method === "POST") {
        const body = init?.body ? JSON.parse(String(init.body)) : null;
        created.push(body);
        return Promise.resolve(jsonResponse(entry({ id: created.length })));
      }
      throw new Error(`Unexpected fetch: ${url} (${method})`);
    });

    render(<GlossaryEditor tenantId="acme" />);
    await waitFor(() => {
      expect(screen.getByTestId("glossary-empty")).toBeInTheDocument();
    });

    const csv = [
      "source_term,target_term,source_lang,target_lang,case_sensitive,is_regex,priority,notes",
      `"Hello, World","Hola, Mundo",en,es,false,false,100,greet`,
      `"He said ""ok""","Dijo ok",en,es,true,false,50,quoted`,
    ].join("\n");

    // Confirm the CSV parser handles the edge cases first.
    const records = parseCsv(csv);
    expect(records).toHaveLength(2);
    expect(records[0].source_term).toBe("Hello, World");
    expect(records[1].source_term).toBe('He said "ok"');

    const file = new File([csv], "g.csv", { type: "text/csv" });
    const fileInput = screen.getByTestId("glossary-import-input") as HTMLInputElement;
    Object.defineProperty(fileInput, "files", { value: [file] });
    fireEvent.change(fileInput);

    await waitFor(() => {
      expect(created).toHaveLength(2);
    });
    expect(created[0]).toMatchObject({
      source_term: "Hello, World",
      target_term: "Hola, Mundo",
      source_lang: "en",
      target_lang: "es",
      case_sensitive: false,
    });
    expect(created[1]).toMatchObject({
      source_term: 'He said "ok"',
      case_sensitive: true,
    });
    expect(screen.getByTestId("glossary-import-status")).toHaveTextContent(
      /Imported 2/
    );
  });

  it("creates a new entry via the modal", async => {
    let postBody: unknown = null;
    let listCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.includes("/glossary") && !url.match(/\/glossary\/\d+/) && method === "GET") {
        listCount += 1;
        const e =
          listCount === 1
            ? []
            : [entry({ id: 11, source_term: "alpha", target_term: "alfa" })];
        return Promise.resolve(
          jsonResponse({ entries: e, total: e.length, page: 1, page_size: 100 })
        );
      }
      if (url.includes("/glossary") && method === "POST") {
        postBody = init?.body ? JSON.parse(String(init.body)) : null;
        return Promise.resolve(
          jsonResponse(entry({ id: 11, source_term: "alpha", target_term: "alfa" }))
        );
      }
      throw new Error(`Unexpected fetch: ${url} (${method})`);
    });

    render(<GlossaryEditor tenantId="acme" />);
    await waitFor(() => {
      expect(screen.getByTestId("glossary-empty")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("glossary-add-button"));
    fireEvent.change(screen.getByTestId("glossary-create-source-term"), {
      target: { value: "alpha" },
    });
    fireEvent.change(screen.getByTestId("glossary-create-target-term"), {
      target: { value: "alfa" },
    });
    fireEvent.click(screen.getByTestId("glossary-create-submit"));

    await waitFor(() => {
      expect(postBody).toMatchObject({
        source_term: "alpha",
        target_term: "alfa",
        source_lang: "en",
        target_lang: "es",
      });
    });
  });
});
// Avoid unused-import warnings under noUnusedLocals.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const _unused = noContent;
