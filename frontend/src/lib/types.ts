/**
 * Shared type definitions for the EDCOCR operator console.
 *
 * These mirror the FastAPI Pydantic models in `api/models.py` (see
 * JobStatusResponse, JobListResponse, JobResultResponse), the WebSocket
 * frame shapes in `api/routers/ws.py`, and the dataclass `to_dict()`
 * shapes in `api/dashboard.py` and `api/fleet_status.py`.
 *
 * Only the fields the UI consumes are typed; unknown fields are tolerated.
 */

// ---------------------------------------------------------------------------
// Jobs (D3)
// ---------------------------------------------------------------------------

export type JobStatus =
  | "submitted"
  | "queued"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled"
  | string;

export interface JobProgress {
  total_pages: number;
  pages_completed: number;
  percent_complete: number;
  current_stage: string;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  priority: string;
  source_file: string;
  progress: JobProgress | null;
  settings: Record<string, unknown>;
  webhook_status: string | null;
}

export interface JobListResponse {
  jobs: Job[];
  total: number;
  limit: number;
  offset: number;
  page?: number | null;
  per_page?: number | null;
}

export interface JobResult {
  job_id: string;
  status: JobStatus;
  completed_at: string | null;
  processing_time_seconds: number | null;
  artifacts: Record<string, string>;
  metadata: Record<string, unknown>;
}

export interface OutputArtifact {
  output_type: string;
  filename: string;
  relative_path: string;
  size_bytes: number;
  mime_type: string;
  schema_version: string;
}

export interface OutputManifest {
  job_id: string;
  artifacts: OutputArtifact[];
  schema_versions: Record<string, string>;
}

export interface JobPage {
  page_number: number;
  status: string;
  confidence: number | null;
  language: string | null;
  fallback: string | null;
}

/**
 * WebSocket message frames from `api/routers/ws.py`.
 * The server sends discriminated objects keyed by `type`.
 */
export type JobWSMessage =
  | { type: "connected"; job_id: string; status: JobStatus }
  | {
      type: "progress";
      job_id: string;
      status: JobStatus;
      pages_completed?: number;
      total_pages?: number;
      percent?: number;
      current_stage?: string;
    }
  | { type: "completed"; job_id: string; status: "completed"; output_path: string }
  | { type: "failed"; job_id: string; status: "failed"; error: string }
  | { type: "cancelled"; job_id: string; status: "cancelled" }
  | { type: "error"; message: string }
  | { type: "pong" };

export type WSStatus =
  | "idle"
  | "connecting"
  | "open"
  | "authenticating"
  | "closed"
  | "reconnecting"
  | "error";

export interface JobsListFilters {
  status?: string;
  search?: string;
  start_date?: string;
  end_date?: string;
  limit: number;
  offset: number;
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

export interface BatchJobSummary {
  job_id: string;
  source_file: string;
  status: JobStatus;
}

export interface BatchProgressInfo {
  submitted: number;
  processing: number;
  completed: number;
  failed: number;
  cancelled: number;
  percent_complete: number;
}

export interface BatchStatusResponse {
  batch_id: string;
  status: JobStatus;
  created_at: string;
  completed_at: string | null;
  processing_time: number | null;
  total_jobs: number;
  progress: BatchProgressInfo;
  jobs: BatchJobSummary[];
  settings: Record<string, unknown>;
  webhook_status: string | null;
}

export interface BatchSubmitResponse {
  batch_id: string;
  status: JobStatus;
  created_at: string;
  total_jobs: number;
  priority: string;
  jobs: BatchJobSummary[];
  links?: Record<string, string>;
}

export interface BatchListResponse {
  batches: BatchStatusResponse[];
  total: number;
  limit: number;
  offset: number;
}

// ---------------------------------------------------------------------------
// Health (D2)
// ---------------------------------------------------------------------------

export type HealthOverall = "healthy" | "degraded" | "unhealthy" | string;

export interface SubsystemCheck {
  status: "healthy" | "degraded" | "unhealthy" | string;
  message?: string;
  latency_ms?: number | null;
}

export interface DetailedHealthResponse {
  status: HealthOverall;
  version: string;
  uptime_seconds: number;
  jobs: Record<string, number>;
  checks?: Record<string, SubsystemCheck>;
}

// ---------------------------------------------------------------------------
// Dashboard snapshot (api/dashboard.py DashboardSnapshot.to_dict)
// ---------------------------------------------------------------------------

export interface DashboardThroughput {
  pages_per_minute: number;
  docs_per_hour: number;
  bytes_per_second: number;
}

export interface DashboardLatency {
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface DashboardJobCounts {
  total: number;
  active: number;
  completed: number;
  failed: number;
  queued: number;
}

export interface DashboardSnapshot {
  timestamp: number;
  throughput: DashboardThroughput;
  latency: DashboardLatency;
  jobs: DashboardJobCounts;
  stages?: Array<Record<string, unknown>>;
  tenant_id?: string;
}

// ---------------------------------------------------------------------------
// Fleet snapshot (api/fleet_status.py FleetSnapshot.to_dict)
// ---------------------------------------------------------------------------

export interface FleetSummary {
  total_workers: number;
  online: number;
  busy: number;
  idle: number;
  offline: number;
  error: number;
  draining: number;
}

export interface FleetGpu {
  total_gpus: number;
  avg_utilization_pct: number;
  avg_memory_pct: number;
  total_memory_mb: number;
  used_memory_mb: number;
}

export type WorkerStatus =
  | "online"
  | "busy"
  | "idle"
  | "offline"
  | "draining"
  | "error";

export interface GpuInfo {
  gpu_id: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_free_mb: number;
  memory_utilization_pct: number;
  utilization_pct: number;
  temperature_c: number;
}

export interface Worker {
  worker_id: string;
  hostname: string;
  state: WorkerStatus;
  capabilities: string[];
  gpus: GpuInfo[];
  current_job_id: string;
  jobs_completed: number;
  jobs_failed: number;
  uptime_seconds: number;
  last_heartbeat: number;
  is_healthy: boolean;
  queue_name: string;
}

export interface FleetSnapshot {
  timestamp: number;
  summary: FleetSummary;
  gpu: FleetGpu;
  workers?: Worker[];
}

// ---------------------------------------------------------------------------
// Queue snapshot (api/routers/alerts.py QueueSnapshot.to_dict, D5)
// ---------------------------------------------------------------------------

export interface Queue {
  queue_name: string;
  depth: number;
  warning_threshold: number | null;
  critical_threshold: number | null;
  /** Optional fields when the API includes them in future revisions. */
  consumers?: number;
  in_flight?: number;
  oldest_item_age_seconds?: number;
  warning_wait_seconds?: number | null;
  critical_wait_seconds?: number | null;
}

export interface QueueThreshold {
  queue_name: string;
  warning_depth: number;
  critical_depth: number;
  warning_wait_seconds: number;
  critical_wait_seconds: number;
}

export type AlertSeverity = "info" | "warning" | "critical";
export type AlertState = "active" | "acknowledged" | "resolved";

export interface QueueAlert {
  alert_id: string;
  queue_name: string;
  severity: AlertSeverity;
  state: AlertState;
  message: string;
  triggered_at: number;
  resolved_at: number;
  acknowledged_at: number;
  current_depth: number;
  threshold_value: number;
}

export interface QueueSnapshot {
  timestamp: number;
  total_depth: number;
  queues: Queue[];
  active_alerts: QueueAlert[];
}

// ---------------------------------------------------------------------------
// D6: server-side jobs filter state
// ---------------------------------------------------------------------------

export type JobsSort =
  | "submitted_at_desc"
  | "submitted_at_asc"
  | "duration_desc"
  | "status";

export interface JobsFilterState {
  /** Multi-select status chips. Empty array == "all". */
  status: string[];
  /** ISO 8601 datetime-local string (YYYY-MM-DDTHH:mm) -- inclusive lower bound. */
  submitted_after: string;
  /** ISO 8601 datetime-local string -- inclusive upper bound. */
  submitted_before: string;
  /** Free-text substring against job_id and source_file. */
  q: string;
  /** Sort order applied server-side. */
  sort: JobsSort;
}

// ---------------------------------------------------------------------------
// D7: per-job NDJSON log stream
// ---------------------------------------------------------------------------

export type JobLogLevel = "DEBUG" | "INFO" | "WARN" | "WARNING" | "ERROR";

export interface JobLogRecord {
  ts: string;
  level: JobLogLevel | string;
  code: string;
  job_id: string;
  message: string;
  data?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// D8: human review queue
//
// Mirrors `api/models.py` ReviewItemResponse / ReviewQueueResponse /
// ReviewStatsResponse / ReviewDecisionRequest.
// ---------------------------------------------------------------------------

/**
 * Review item lifecycle. Backend authoritative values:
 *   pending     -- newly enqueued, awaiting human decision
 *   approved    -- human approved the OCR/translation output
 *   rejected    -- human rejected; output discarded or quarantined
 *   reprocess   -- send back through pipeline (a.k.a. "escalated")
 */
export type ReviewStatus = "pending" | "approved" | "rejected" | "reprocess";

/**
 * Decision values that may be posted to /decision. Mirrors the
 * `Literal["approved","rejected","reprocess"]` field in the FastAPI model.
 */
export type ReviewDecision = "approved" | "rejected" | "reprocess";

export interface ReviewItem {
  review_id: string;
  job_id: string;
  reason: string;
  confidence: number;
  quality_classification: string;
  status: ReviewStatus | string;
  reviewer: string;
  decision_notes: string;
  /** ISO 8601 UTC string. Empty string when unset. */
  created_at: string;
  /** ISO 8601 UTC string. Empty string while still pending. */
  reviewed_at: string;
  metadata: Record<string, unknown>;
}

export interface ReviewQueueResponse {
  items: ReviewItem[];
  total: number;
}

export interface ReviewStats {
  pending: number;
  approved: number;
  rejected: number;
  reprocess: number;
  total: number;
  avg_review_seconds: number;
  oldest_pending: string;
}

/** Body shape for POST /api/v1/review/{id}/decision. */
export interface ReviewDecisionRequest {
  status: ReviewDecision;
  reviewer?: string;
  notes?: string;
}

/**
 * Strong-auth method that the operator selects in the certify dialog.
 * The backend custody event records this verbatim.
 */
export type CertifyAuthMethod = "piv_cac" | "oidc_mfa" | "hardware_token";

/** Body shape for POST /api/v1/review/{id}/certify. */
export interface CertifyRequest {
  auth_method: CertifyAuthMethod;
  /** Opaque token / signed assertion supplied by the auth method. */
  auth_token: string;
  notes?: string;
}

export interface ReviewQueueFilters {
  /** Empty array == "all statuses" (UI default shows pending). */
  status: ReviewStatus[];
  /** Optional review reason filter -- maps to the `reason` query param. */
  reason: string;
  /** Free-text substring against job_id (client-side only). */
  q: string;
}

// ---------------------------------------------------------------------------
// D9: Tenant management UI
//
// Mirrors api/routers/translation_admin.py (TenantConfigOut, TenantConfigUpsert,
// GlossaryEntryOut, GlossaryEntryIn, GlossaryEntryUpdate, GlossaryListOut) and
// api/routers/admin.py (TenantResponse).
// ---------------------------------------------------------------------------

export type TenantQualityTier = "draft" | "standard" | "legal";

export interface TenantConfig {
  tenant_id: string;
  /** BCP-47 target language codes the tenant has authorized. */
  target_languages: string[];
  /** Engine identifiers (e.g. "opus_mt", "nllb_200", "madlad_400"). */
  preferred_engines: string[];
  /** Allows NC-licensed engines (NLLB-200) for commercial use. */
  allow_nc_licensed: boolean;
  /** When true, downstream writes flag jobs as needing certified review. */
  require_certified: boolean;
  default_quality_tier: TenantQualityTier;
  created_at?: string | null;
  updated_at?: string | null;
}

export type TenantConfigUpdate = Omit<
  TenantConfig,
  "tenant_id" | "created_at" | "updated_at"
>;

export interface GlossaryEntry {
  id: number;
  tenant_id: string;
  source_term: string;
  target_term: string;
  source_lang: string;
  target_lang: string;
  case_sensitive: boolean;
  is_regex: boolean;
  priority: number;
  notes?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface GlossaryFilters {
  source_lang?: string;
  target_lang?: string;
  page?: number;
  page_size?: number;
}

export interface GlossaryListResponse {
  entries: GlossaryEntry[];
  total: number;
  page: number;
  page_size: number;
}

/** Subset of api/routers/admin.py TenantResponse the management UI consumes. */
export interface TenantSummary {
  tenant_id: string;
  name: string;
  display_name?: string | null;
  status: string;
  tier: string;
  created_at?: string | null;
  updated_at?: string | null;
  max_concurrent_jobs?: number;
  max_pages_per_month?: number;
  max_storage_bytes?: number;
  allowed_features?: string[];
  admin_email?: string | null;
}

// ---------------------------------------------------------------------------
// D10: client-side settings (browser-local, persisted to localStorage)
//
// These are per-browser preferences. There is NO server-side settings store --
// the operator console reads/writes the persisted shape from `settings-store.ts`
// under the key `ocr-local:settings` and migrates older shapes if encountered.
// ---------------------------------------------------------------------------

export type SettingsTheme = "light" | "dark" | "system";
export type SettingsDateFormat = "iso" | "us" | "eu";
export type SettingsPageSize = 10 | 25 | 50 | 100;

export interface SettingsGeneral {
  theme: SettingsTheme;
  /** IANA timezone identifier; default = browser's resolved timezone. */
  timezone: string;
  dateFormat: SettingsDateFormat;
}

export interface SettingsApi {
  /** Empty string falls through to current origin; otherwise must be a valid URL. */
  baseUrl: string;
  /** Request timeout in milliseconds. Range [1000, 120000]. */
  timeoutMs: number;
  /**
   * Redacted preview of the operator API key (first 4 + last 4 of the actual
   * key). The actual key is stored under `ocr_local_api_key` by the auth
   * layer; this field is for display only and must never round-trip a usable
   * credential.
   */
  apiKeyRedactedPreview: string;
}

export interface SettingsDisplay {
  /** Auto-refresh interval in seconds. Range [5, 600]. */
  autoRefreshSeconds: number;
  pageSize: SettingsPageSize;
  compactMode: boolean;
  showAdvancedColumns: boolean;
}

export interface SettingsNotifications {
  desktopEnabled: boolean;
  soundEnabled: boolean;
  onJobComplete: boolean;
  onJobFailure: boolean;
  onReviewItem: boolean;
}

/**
 * Top-level shape persisted under `ocr-local:settings`. The literal-typed
 * `schema_version` lets `migrateSettings()` transition old shapes forward.
 */
export interface Settings {
  schema_version: 1;
  general: SettingsGeneral;
  api: SettingsApi;
  display: SettingsDisplay;
  notifications: SettingsNotifications;
}

// ---------------------------------------------------------------------------
// D11: Alert configuration UI
//
// Mirrors the (not-yet-provisioned) admin endpoints under
// /api/v1/admin/alerts/* and /api/v1/admin/alert-channels/*. The UI degrades
// gracefully on 404/501 (API not provisioned) and 403 (caller lacks
// platform-admin scope).
//
// Note: `AlertSeverity` already exists earlier in this file (queue alerts);
// the existing `"info" | "warning" | "critical"` union is reused. The
// existing `AlertState` ("active" | "acknowledged" | "resolved") is for
// queue alerts and conflicts with the rule-based admin lifecycle, so
// the alert-rule lifecycle is named `AdminAlertState` here.
// ---------------------------------------------------------------------------

/**
 * Lifecycle state of an alert as exposed by the admin alerts API.
 *
 *   firing   -- threshold currently breached
 *   pending  -- threshold breached but evaluation window not yet elapsed
 *   inactive -- not firing; rule is enabled and idle
 *   muted    -- temporarily suppressed by an operator
 */
export type AdminAlertState = "firing" | "pending" | "inactive" | "muted";

/** Threshold unit hint used to render the appropriate input control. */
export type AlertThresholdUnit = "bytes" | "seconds" | "count" | "percent";

/** Notification channel kind. Targets are partially redacted in list views. */
export type NotificationChannelType = "webhook" | "slack" | "email";

/**
 * A currently-firing (or recently-firing) alert instance.
 */
export interface Alert {
  id: string;
  rule_id: string;
  severity: AlertSeverity;
  state: AdminAlertState;
  tenant_id?: string | null;
  /** ISO 8601 UTC. */
  started_at: string;
  /** ISO 8601 UTC. */
  last_seen: string;
  message: string;
  labels: Record<string, string>;
}

/**
 * Configuration of an alert rule. The PromQL `expression` is read-only in
 * the UI; only the operator-tunable subset (`AlertRuleUpdate`) is editable.
 */
export interface AlertRule {
  id: string;
  name: string;
  severity: AlertSeverity;
  /** PromQL expression (read-only in this UI). */
  expression: string;
  threshold_value: number;
  threshold_unit: AlertThresholdUnit;
  /** "for" duration before pending flips to firing. */
  evaluation_window_seconds: number;
  enabled: boolean;
  /** Channel ids the rule fans out to. */
  notification_channels: string[];
  /** ISO 8601 of the most recent firing. Null/empty when never fired. */
  last_triggered_at?: string | null;
  current_state: AdminAlertState;
  /** Optional human-readable description / runbook link. */
  description?: string | null;
}

/**
 * Subset of fields the operator may PATCH on a rule. `expression` is
 * intentionally absent -- PromQL changes go through git, not this UI.
 */
export interface AlertRuleUpdate {
  threshold_value?: number;
  evaluation_window_seconds?: number;
  severity?: AlertSeverity;
  enabled?: boolean;
  notification_channels?: string[];
}

/**
 * A notification destination (webhook URL, Slack channel, email recipient
 * list). The `target` is expected to come back pre-redacted.
 */
export interface NotificationChannel {
  id: string;
  type: NotificationChannelType;
  /** Pre-redacted target string (e.g. "slack:#ops-***l"). */
  target: string;
  enabled: boolean;
  last_test_at?: string | null;
  last_test_ok?: boolean | null;
}

/** Body shape for POST /api/v1/admin/alerts/{id}/mute. */
export interface AlertMuteRequest {
  reason: string;
  /** Optional duration in seconds; backend may default. */
  duration_seconds?: number;
}

// ---------------------------------------------------------------------------
// D12: feature flag toggle UI
//
// Mirrors `ocr_local/config/feature_flags.py` plus a backend admin layer that
// the UI assumes lives at `/api/v1/admin/feature-flags/*`. The UI never
// directly mutates flags -- mutations go through a custody-logged change
// request reviewed by platform operators.
// ---------------------------------------------------------------------------

/**
 * Where the current value of a flag is being sourced from. The badge color
 * scheme in `FeatureFlagsList` reflects this:
 *   env       -> blue   (process environment override at startup)
 *   config    -> green  (configuration file / PipelineConfig dataclass)
 *   database  -> purple (per-tenant or runtime DB row)
 *   default   -> gray   (no override -- code-level default)
 */
export type FeatureFlagSource = "env" | "config" | "database" | "default";

/**
 * Logical grouping for sidebar/section rendering. Categories collapse/expand
 * independently; preference is persisted to localStorage under
 * `ocr-local:flags-ui`.
 */
export type FeatureFlagCategory =
  | "translation"
  | "custody"
  | "pipeline"
  | "experimental"
  | "operations";

/**
 * Discriminator for the runtime value type. Drives input rendering inside
 * the change-request dialog and pill styling in the list view.
 */
export type FeatureFlagValueType = "boolean" | "string" | "enum" | "integer";

/**
 * Lifecycle of a flag-change request. Backend authoritative values:
 *   pending      -- awaiting platform-admin approval
 *   approved     -- approved but not yet applied (waiting bake or rollout)
 *   rejected     -- denied by reviewer
 *   applied      -- live in the running config
 *   rolled_back  -- previously applied, then reverted
 */
export type FeatureFlagRequestStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "applied"
  | "rolled_back";

/**
 * A single feature flag definition + its current effective value.
 *
 * `current_value` is union-typed because the backend can return any primitive
 * shape. Consumers should branch on `value_type` before assuming a concrete
 * type.
 */
export interface FeatureFlag {
  key: string;
  category: FeatureFlagCategory | string;
  value_type: FeatureFlagValueType;
  current_value: boolean | string | number | null;
  default_value: boolean | string | number | null;
  source: FeatureFlagSource;
  description: string;
  /**
   * Documented soak window. When > 0 and the flag last changed inside the
   * window, the detail page must show a warning banner. The translation
   * gates (gotcha #86) use 48h.
   */
  requires_bake_hours?: number;
  /** When true, change-request requires a strong-auth (PIV/CAC, OIDC+MFA, FIDO2). */
  requires_strong_auth: boolean;
  /** Allowed values for `value_type === "enum"`. */
  allowed_values?: Array<string | number>;
  /** ISO 8601 UTC string of the most recent applied change, if any. */
  last_changed_at?: string | null;
}

/**
 * One row in a flag's audit trail. Mirrors what the backend would record on
 * the custody chain when a change request is filed, approved, applied, or
 * rolled back.
 */
export interface FeatureFlagHistoryEntry {
  request_id: string;
  flag_key: string;
  previous_value: boolean | string | number | null;
  new_value: boolean | string | number | null;
  reason: string;
  requested_by: string;
  approved_by?: string | null;
  requested_at: string;
  applied_at?: string | null;
  status: FeatureFlagRequestStatus;
}

/**
 * Body shape for POST /api/v1/admin/feature-flags/{key}/change-request.
 * Re-uses `CertifyAuthMethod` from D8 because the backend treats the auth
 * payload identically -- both flows write a strong-auth custody event.
 */
export interface FlagChangeRequest {
  flag_key: string;
  new_value: boolean | string | number | null;
  reason: string;
  auth_method?: CertifyAuthMethod;
  auth_token?: string;
}

