/**
 * Feature-flag admin API helpers (D12).
 *
 * Mirrors a backend admin surface assumed to live at
 * `/api/v1/admin/feature-flags/*`. The UI never PUT/PATCHes a flag value
 * directly -- mutations go through `submitChangeRequest` which the backend
 * routes through a custody-logged approval workflow (gotcha #87: emit the
 * rejection custody event BEFORE raising 403).
 *
 * Backed by `lib/api-client.ts:get/post`. We intentionally route POST through
 * the shared client so 401/403 still clear the cached API key.
 */

import { ApiError, get, post } from "./api-client";
import type {
  FeatureFlag,
  FeatureFlagHistoryEntry,
  FlagChangeRequest,
} from "./types";

interface ListResponse {
  flags: FeatureFlag[];
}

interface HistoryResponse {
  entries: FeatureFlagHistoryEntry[];
  total: number;
}

/**
 * List every known flag with its current value, default, and source.
 *
 * The backend may return either a bare array or `{ flags: [...] }` -- we
 * normalize to the array shape so callers don't have to branch.
 */
export async function listFlags(signal?: AbortSignal): Promise<FeatureFlag[]> {
  const raw = await get<ListResponse | FeatureFlag[]>(
    "/api/v1/admin/feature-flags",
    { signal }
  );
  if (Array.isArray(raw)) return raw;
  return raw.flags ?? [];
}

/**
 * Fetch a single flag by key. Returns null on 404 so callers can surface a
 * "not found" empty state without a thrown rejection.
 */
export async function getFlag(
  key: string,
  signal?: AbortSignal
): Promise<FeatureFlag | null> {
  try {
    return await get<FeatureFlag>(
      `/api/v1/admin/feature-flags/${encodeURIComponent(key)}`,
      { signal }
    );
  } catch (err: unknown) {
    if (err instanceof ApiError && err.status === 404) {
      return null;
    }
    throw err;
  }
}

/**
 * Fetch the most recent change history for a flag. Backend should return at
 * most ~20 entries; we don't enforce that client-side.
 */
export async function getFlagHistory(
  key: string,
  signal?: AbortSignal
): Promise<FeatureFlagHistoryEntry[]> {
  const raw = await get<HistoryResponse | FeatureFlagHistoryEntry[]>(
    `/api/v1/admin/feature-flags/${encodeURIComponent(key)}/history`,
    { signal }
  );
  if (Array.isArray(raw)) return raw;
  return raw.entries ?? [];
}

/**
 * Submit a change request. On 202 Accepted the backend has filed a pending
 * custody event; on 403 with `error_code: "strong_auth_required"` the dialog
 * must surface a method-specific error message.
 *
 * The backend is the only authority on whether a flip is allowed -- the UI
 * MUST NEVER pretend the request succeeded just because the form was filled
 * out.
 */
export async function submitChangeRequest(
  key: string,
  payload: FlagChangeRequest,
  signal?: AbortSignal
): Promise<FeatureFlagHistoryEntry> {
  return post<FeatureFlagHistoryEntry>(
    `/api/v1/admin/feature-flags/${encodeURIComponent(key)}/change-request`,
    payload,
    { signal }
  );
}
