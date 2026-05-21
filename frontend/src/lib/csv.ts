/**
 * Minimal CSV parser/serializer used by the glossary import/export flow.
 *
 * Goals:
 *  - Zero dependencies (no Papa, no csv-parse).
 *  - RFC 4180 friendly: quoted fields, embedded commas, embedded quotes
 *    represented as a doubled `""`, CRLF line endings.
 *  - BOM-tolerant: a leading `﻿` is stripped.
 *  - Header row required for `parseCsv`; `serializeCsv` always emits one.
 *
 * Out of scope: streaming, custom delimiters, escape characters other than
 * `"`, comment lines.
 */

const BOM = "﻿";

/**
 * Parse a CSV document with a header row into objects keyed by header name.
 *
 * @param text Raw CSV text (BOM-tolerant; CRLF and LF accepted).
 * @returns A list of records. The first row is treated as the header.
 * @throws when a quoted field is unterminated.
 */
export function parseCsv(text: string): Array<Record<string, string>> {
  if (typeof text !== "string") return [];
  const stripped = text.startsWith(BOM) ? text.slice(BOM.length) : text;
  const rows = parseCsvRows(stripped);
  if (rows.length === 0) return [];
  const header = rows[0].map((h) => h.trim());
  const records: Array<Record<string, string>> = [];
  for (let i = 1; i < rows.length; i++) {
    const row = rows[i];
    // Skip fully blank lines (common when files end with a newline).
    if (row.length === 1 && row[0] === "") continue;
    const record: Record<string, string> = {};
    for (let c = 0; c < header.length; c++) {
      record[header[c]] = c < row.length ? row[c] : "";
    }
    records.push(record);
  }
  return records;
}

/**
 * Tokenize CSV text into rows of fields.
 *
 * Exposed for tests; most callers want `parseCsv`.
 */
export function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  let i = 0;
  const len = text.length;

  while (i < len) {
    const ch = text[i];

    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < len && text[i + 1] === '"') {
          // Escaped quote inside quoted field.
          field += '"';
          i += 2;
          continue;
        }
        // Closing quote.
        inQuotes = false;
        i += 1;
        continue;
      }
      field += ch;
      i += 1;
      continue;
    }

    // Outside quotes.
    if (ch === '"') {
      // Opening quote (only valid at start of a field, but we're permissive).
      inQuotes = true;
      i += 1;
      continue;
    }
    if (ch === ",") {
      row.push(field);
      field = "";
      i += 1;
      continue;
    }
    if (ch === "\r") {
      // Treat CRLF as a single line break.
      if (i + 1 < len && text[i + 1] === "\n") {
        i += 2;
      } else {
        i += 1;
      }
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }
    if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      i += 1;
      continue;
    }
    field += ch;
    i += 1;
  }

  if (inQuotes) {
    throw new Error("Unterminated quoted field in CSV input");
  }

  // Flush trailing field/row if the file did not end with a newline.
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const str = String(value);
  // Quote when the field contains comma, quote, CR, or LF.
  if (/[",\r\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export interface SerializeCsvOptions {
  /** Include the UTF-8 BOM for Excel compatibility. Default: false. */
  bom?: boolean;
}

/**
 * Serialize an array of records into CSV text with a header row derived from
 * the union of keys (or, if `header` supplied, the explicit column order).
 */
export function serializeCsv<T extends Record<string, unknown>>(
  records: T[],
  header?: string[],
  options?: SerializeCsvOptions
): string {
  const cols =
    header ??
    Array.from(
      records.reduce<Set<string>>((acc, r) => {
        for (const k of Object.keys(r)) acc.add(k);
        return acc;
      }, new Set<string>())
    );
  const lines: string[] = [];
  lines.push(cols.map((c) => csvEscape(c)).join(","));
  for (const r of records) {
    lines.push(cols.map((c) => csvEscape(r[c])).join(","));
  }
  const body = lines.join("\r\n") + "\r\n";
  return options?.bom ? BOM + body : body;
}

/**
 * Convenience helper: trigger a browser download for a string payload.
 *
 * No-ops in non-browser runtimes (so unit tests can import without crashing).
 */
export function triggerDownload(filename: string, content: string, mimeType = "text/csv"): void {
  if (typeof window === "undefined" || typeof document === "undefined") return;
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Defer revocation to ensure the click is processed.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
