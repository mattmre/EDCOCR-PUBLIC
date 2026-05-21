/**
 * Client-side hash-chain verification for forensic custody logs.
 *
 * The producer side of the chain lives at ``ocr_local/features/custody.py``.
 * Each event is a JSON object with the following fields:
 *
 *   {
 *     "document_id":  string,
 *     "event_type":   string,
 *     "timestamp":    ISO-8601 string with millisecond precision and UTC ("Z") suffix,
 *     "data":         object (event-specific payload),
 *     "prev_hash":    string | null,   // SHA-256 hex of previous event, null on first
 *     "hash":         string           // SHA-256 hex of {everything above}
 *   }
 *
 * The recorded ``hash`` is computed by the producer as:
 *
 *   sha256(json.dumps(event_without_hash, sort_keys=True, default=str).encode("utf-8"))
 *
 * To match the producer here we must:
 *   1. drop the ``hash`` field,
 *   2. emit the remaining fields with **sorted keys** (lexicographic by key name),
 *   3. use Python ``json.dumps`` defaults: ASCII-only output, no whitespace
 *      between separators except the ", " / ": " pair, and Python escape
 *      semantics for control characters,
 *   4. SHA-256 the UTF-8 bytes.
 *
 * This file implements a minimal canonical JSON encoder that matches Python's
 * ``json.dumps(..., sort_keys=True)`` for the value shapes that custody events
 * actually contain (str, number, bool, null, dict, list). Anything richer
 * (dates, sets, custom classes) hits the ``default=str`` fallback on the
 * Python side and is already a string by the time it reaches the JSONL log,
 * so we only need to faithfully encode primitives + nested arrays/objects.
 *
 * The forensic schema reference is ``docs/forensic/custody-schema-v1.json``.
 * The schema there is the *intended* v1 envelope (with ``chain_hash`` /
 * ``event_id``); the live producer in custody.py uses the simpler
 * ``prev_hash`` / ``hash`` envelope. This module verifies the live producer
 * format. If the schema-v1 envelope ships, ``canonicalEventBytes`` and
 * ``recomputeEventHash`` will need to be updated together with the producer.
 */

export interface CustodyEvent {
  document_id: string;
  event_type: string;
  timestamp: string;
  data: Record<string, unknown>;
  prev_hash: string | null;
  hash: string;
  // Allow upstream to add fields without breaking verification; canonical
  // serialization picks them up via Object.keys() iteration after sort.
  [extra: string]: unknown;
}

export type VerificationStatus = "intact" | "broken" | "verifying" | "empty";

export interface VerificationResult {
  status: VerificationStatus;
  totalEvents: number;
  verifiedEvents: number;
  /** Index of the first event whose recorded hash didn't match the recomputed hash. */
  brokenAtIndex?: number;
  /** Hash field of the first broken event (truncated for display caller responsibility). */
  brokenAtEventHash?: string;
  /** Human-readable reason describing the break. */
  reason?: string;
}

// ---------------------------------------------------------------------------
// Canonical JSON serialization (matches Python json.dumps(sort_keys=True))
// ---------------------------------------------------------------------------

function encodeString(value: string): string {
  // Python json.dumps default is ensure_ascii=True. Mirror that: any
  // non-ASCII code point is emitted as \uXXXX. ASCII control characters
  // and the standard escapees follow the same table as JS JSON.stringify.
  let out = '"';
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code === 0x22) {
      out += '\\"';
    } else if (code === 0x5c) {
      out += "\\\\";
    } else if (code === 0x08) {
      out += "\\b";
    } else if (code === 0x09) {
      out += "\\t";
    } else if (code === 0x0a) {
      out += "\\n";
    } else if (code === 0x0c) {
      out += "\\f";
    } else if (code === 0x0d) {
      out += "\\r";
    } else if (code < 0x20 || code > 0x7e) {
      out += "\\u" + code.toString(16).padStart(4, "0");
    } else {
      out += value[i];
    }
  }
  out += '"';
  return out;
}

function encodeNumber(value: number): string {
  if (!Number.isFinite(value)) {
    // Python emits NaN / Infinity by default; treat as encoding error
    // because the producer would never write one to the custody log.
    throw new Error(`Cannot canonicalize non-finite number: ${value}`);
  }
  if (Number.isInteger(value)) {
    return value.toString();
  }
  // For floats Python's repr-based formatting differs subtly from JS, but
  // custody payloads rarely carry floats; when they do they pass through
  // ``default=str`` (already a string) or are integers. Use toString().
  return value.toString();
}

export function canonicalize(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (value === undefined) {
    // Python json.dumps cannot encode undefined; treat as null for safety.
    return "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return encodeNumber(value);
  }
  if (typeof value === "string") {
    return encodeString(value);
  }
  if (Array.isArray(value)) {
    const parts = value.map((item) => canonicalize(item));
    return "[" + parts.join(", ") + "]";
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const parts = keys.map((k) => {
      return encodeString(k) + ": " + canonicalize(obj[k]);
    });
    return "{" + parts.join(", ") + "}";
  }
  // Functions / symbols / bigints are not part of the custody schema.
  throw new Error(`Cannot canonicalize value of type ${typeof value}`);
}

export function canonicalEventBytes(event: CustodyEvent): Uint8Array {
  const { hash: _omit, ...rest } = event;
  void _omit;
  const json = canonicalize(rest);
  return new TextEncoder().encode(json);
}

// ---------------------------------------------------------------------------
// SHA-256 (Web Crypto in the browser, Node fallback for tests)
// ---------------------------------------------------------------------------

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const subtle =
    typeof globalThis !== "undefined" &&
    typeof globalThis.crypto !== "undefined" &&
    globalThis.crypto.subtle
      ? globalThis.crypto.subtle
      : null;
  if (subtle) {
    // The bytes view originates from TextEncoder.encode whose buffer is
    // always a real ArrayBuffer, but the latest TS lib types now widen
    // Uint8Array to ArrayBufferLike. Pass through .buffer to satisfy
    // SubtleCrypto.digest's BufferSource parameter.
    const digest = await subtle.digest("SHA-256", bytes.buffer as ArrayBuffer);
    return bufferToHex(new Uint8Array(digest));
  }
  // jsdom in older Node releases may not expose crypto.subtle. Fall back to
  // node's crypto module only when subtle is unavailable. The production
  // browser path never hits this branch. The dynamic specifier is hidden
  // from webpack's static analyzer (it would otherwise try to bundle
  // node:crypto for the client and fail with UnhandledSchemeError).
  const moduleName = "node:" + "crypto";
  const dynamicImport = new Function("m", "return import(m)") as (
    m: string) => Promise<typeof import("crypto")>;
  const nodeCrypto = await dynamicImport(moduleName);
  return nodeCrypto.createHash("sha256").update(bytes).digest("hex");
}

function bufferToHex(buffer: Uint8Array): string {
  let out = "";
  for (let i = 0; i < buffer.length; i++) {
    out += buffer[i].toString(16).padStart(2, "0");
  }
  return out;
}

export async function recomputeEventHash(event: CustodyEvent): Promise<string> {
  return sha256Hex(canonicalEventBytes(event));
}

// ---------------------------------------------------------------------------
// Chain verification
// ---------------------------------------------------------------------------

export async function verifyChain(events: CustodyEvent[]): Promise<VerificationResult> {
  if (events.length === 0) {
    return { status: "empty", totalEvents: 0, verifiedEvents: 0 };
  }

  let prevHash: string | null = null;
  for (let i = 0; i < events.length; i++) {
    const event = events[i];

    // 1. prev_hash linkage check
    if (event.prev_hash !== prevHash) {
      return {
        status: "broken",
        totalEvents: events.length,
        verifiedEvents: i,
        brokenAtIndex: i,
        brokenAtEventHash: event.hash,
        reason: `Broken prev_hash at event ${i}: expected ${prevHash ?? "null"}, got ${event.prev_hash ?? "null"}`,
      };
    }

    // 2. hash recompute check
    const computed = await recomputeEventHash(event);
    if (computed !== event.hash) {
      return {
        status: "broken",
        totalEvents: events.length,
        verifiedEvents: i,
        brokenAtIndex: i,
        brokenAtEventHash: event.hash,
        reason: `Tampered event ${i}: recomputed hash ${computed} does not match recorded ${event.hash}`,
      };
    }

    prevHash = event.hash;
  }

  return {
    status: "intact",
    totalEvents: events.length,
    verifiedEvents: events.length,
  };
}
