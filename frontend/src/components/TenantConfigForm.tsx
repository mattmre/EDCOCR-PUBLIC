"use client";

import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cn";
import type {
  TenantConfig,
  TenantConfigUpdate,
  TenantQualityTier,
} from "@/lib/types";

const ALL_ENGINES: Array<{ id: string; label: string; license: string }> = [
  { id: "opus_mt", label: "OPUS-MT", license: "Apache-2.0" },
  { id: "nllb_200", label: "NLLB-200", license: "CC-BY-NC-4.0" },
  { id: "madlad_400", label: "MADLAD-400", license: "Apache-2.0" },
];

const QUALITY_TIERS: TenantQualityTier[] = ["draft", "standard", "legal"];

/** Conservative BCP-47 subset used by the EDCOCR language registry. */
const BCP47_REGEX = /^[A-Za-z]{2,3}(-[A-Za-z]{2,4})?(-[A-Za-z]{2,3})?$/;

const DEFAULT_BCP47_SUGGESTIONS = [
  "en",
  "fr",
  "de",
  "es",
  "it",
  "pt",
  "nl",
  "ru",
  "uk",
  "ja",
  "ko",
  "zh-Hans",
  "zh-Hant",
  "ar",
  "he",
  "hi",
  "tr",
  "vi",
  "pl",
  "cs",
  "sv",
  "fi",
  "da",
  "no",
  "el",
  "fa",
];

export interface TenantConfigFormProps {
  tenantId: string;
  initial: TenantConfig | null;
  onSubmit: (payload: TenantConfigUpdate) => Promise<void> | void;
  /** Optional submit-error to surface above the form. */
  submitError?: string | null;
  saving?: boolean;
  bcp47Suggestions?: string[];
}

interface FieldErrors {
  target_languages?: string;
  preferred_engines?: string;
  default_quality_tier?: string;
}

function defaultUpdate(): TenantConfigUpdate {
  return {
    target_languages: [],
    preferred_engines: [],
    allow_nc_licensed: false,
    require_certified: false,
    default_quality_tier: "standard",
  };
}

export function TenantConfigForm({
  tenantId,
  initial,
  onSubmit,
  submitError,
  saving = false,
  bcp47Suggestions = DEFAULT_BCP47_SUGGESTIONS,
}: TenantConfigFormProps) {
  const [draft, setDraft] = useState<TenantConfigUpdate>(() =>
    initial ? toDraft(initial) : defaultUpdate()
  );
  const [newLang, setNewLang] = useState<string>("");
  const [errors, setErrors] = useState<FieldErrors>({});

  // Re-seed when the upstream tenant or initial config switches identity.
  useEffect(() => {
    setDraft(initial ? toDraft(initial) : defaultUpdate());
    setErrors({});
    setNewLang("");
  }, [tenantId, initial]);

  const dirty = useMemo(() => {
    const baseline = initial ? toDraft(initial) : defaultUpdate();
    return JSON.stringify(baseline) !== JSON.stringify(draft);
  }, [draft, initial]);

  function validate(d: TenantConfigUpdate): FieldErrors {
    const errs: FieldErrors = {};
    for (const lang of d.target_languages) {
      if (!BCP47_REGEX.test(lang)) {
        errs.target_languages = `"${lang}" is not a valid BCP-47 tag`;
        break;
      }
    }
    if (!QUALITY_TIERS.includes(d.default_quality_tier)) {
      errs.default_quality_tier = "Choose a quality tier";
    }
    return errs;
  }

  function handleAddLang() {
    const trimmed = newLang.trim();
    if (!trimmed) return;
    if (!BCP47_REGEX.test(trimmed)) {
      setErrors((e) => ({ ...e, target_languages: `"${trimmed}" is not a valid BCP-47 tag` }));
      return;
    }
    if (draft.target_languages.includes(trimmed)) {
      setNewLang("");
      return;
    }
    setDraft((d) => ({ ...d, target_languages: [...d.target_languages, trimmed] }));
    setErrors((e) => ({ ...e, target_languages: undefined }));
    setNewLang("");
  }

  function handleRemoveLang(lang: string) {
    setDraft((d) => ({
      ...d,
      target_languages: d.target_languages.filter((l) => l !== lang),
    }));
  }

  function toggleEngine(id: string) {
    setDraft((d) => {
      const has = d.preferred_engines.includes(id);
      return {
        ...d,
        preferred_engines: has
          ? d.preferred_engines.filter((e) => e !== id)
          : [...d.preferred_engines, id],
      };
    });
  }

  function handleAllowNcChange(next: boolean) {
    setDraft((d) => {
      // If we are turning OFF nc-licensed, drop NLLB from the engine list.
      const nextEngines = next
        ? d.preferred_engines
        : d.preferred_engines.filter((e) => e !== "nllb_200");
      return { ...d, allow_nc_licensed: next, preferred_engines: nextEngines };
    });
  }

  async function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const errs = validate(draft);
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;
    await onSubmit(draft);
  }

  function handleDiscard() {
    setDraft(initial ? toDraft(initial) : defaultUpdate());
    setErrors({});
    setNewLang("");
  }

  return (
    <form
      onSubmit={handleSave}
      className="space-y-6"
      data-testid="tenant-config-form"
      noValidate
    >
      {submitError ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
          data-testid="tenant-config-error"
        >
          {submitError}
        </div>
      ) : null}

      <fieldset className="space-y-3">
        <legend className="text-sm font-semibold">Target languages (BCP-47)</legend>
        <p className="text-xs text-muted-foreground">
          Translation outputs are restricted to this set. Use codes like{" "}
          <code className="rounded bg-muted px-1">en</code>,{" "}
          <code className="rounded bg-muted px-1">zh-Hans</code>, or{" "}
          <code className="rounded bg-muted px-1">sr-Latn</code>.
        </p>
        <div className="flex flex-wrap gap-2" data-testid="target-languages-chips">
          {draft.target_languages.length === 0 ? (
            <span className="text-xs text-muted-foreground">No target languages set</span>
          ) : null}
          {draft.target_languages.map((lang) => (
            <span
              key={lang}
              className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-xs"
            >
              {lang}
              <button
                type="button"
                aria-label={`Remove ${lang}`}
                onClick={() => handleRemoveLang(lang)}
                className="text-muted-foreground hover:text-foreground"
              >
                ×
              </button>
            </span>
          ))}
        </div>
        <div className="flex gap-2">
          <Input
            list="bcp47-suggestions"
            value={newLang}
            onChange={(e) => setNewLang(e.target.value)}
            placeholder="e.g. en"
            aria-label="Add target language"
            data-testid="target-language-input"
            className="max-w-xs"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleAddLang();
              }
            }}
          />
          <Button
            type="button"
            variant="outline"
            onClick={handleAddLang}
            data-testid="target-language-add"
          >
            Add
          </Button>
        </div>
        <datalist id="bcp47-suggestions">
          {bcp47Suggestions.map((s) => (
            <option value={s} key={s} />
          ))}
        </datalist>
        {errors.target_languages ? (
          <p className="text-xs text-destructive" role="alert">
            {errors.target_languages}
          </p>
        ) : null}
      </fieldset>

      <fieldset className="space-y-2">
        <legend className="text-sm font-semibold">Preferred engines</legend>
        <p className="text-xs text-muted-foreground">
          Order is informational; the router still falls back as engines fail.
        </p>
        <div className="space-y-2" data-testid="engine-list">
          {ALL_ENGINES.map((engine) => {
            const checked = draft.preferred_engines.includes(engine.id);
            const ncBlocked = engine.license === "CC-BY-NC-4.0" && !draft.allow_nc_licensed;
            return (
              <label
                key={engine.id}
                className={cn(
                  "flex items-start gap-2 rounded-md border border-border p-2 text-sm",
                  ncBlocked ? "opacity-60" : ""
                )}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={ncBlocked}
                  onChange={() => toggleEngine(engine.id)}
                  data-testid={`engine-checkbox-${engine.id}`}
                />
                <span className="flex-1">
                  <span className="font-medium">{engine.label}</span>{" "}
                  <span className="text-xs text-muted-foreground">({engine.license})</span>
                  {ncBlocked ? (
                    <span className="ml-2 text-xs text-destructive">
                      Requires non-commercial-use confirmation
                    </span>
                  ) : null}
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>

      <fieldset className="space-y-2">
        <legend className="text-sm font-semibold">Licensing & certification</legend>
        <label className="flex items-start gap-3 rounded-md border border-border p-3 text-sm">
          <input
            type="checkbox"
            checked={draft.allow_nc_licensed}
            onChange={(e) => handleAllowNcChange(e.target.checked)}
            data-testid="allow-nc-toggle"
            className="mt-0.5"
          />
          <span className="space-y-1">
            <span className="block font-medium">Allow NC-licensed engines (e.g. NLLB-200)</span>
            <span className="block text-xs text-muted-foreground" data-testid="allow-nc-help">
              Enables NC-licensed engines for commercial use — confirm operator has secured
              non-commercial-use rights for this tenant. Leave off for commercial deployments
              that have not negotiated a separate license.
            </span>
          </span>
        </label>
        <label className="flex items-start gap-3 rounded-md border border-border p-3 text-sm">
          <input
            type="checkbox"
            checked={draft.require_certified}
            onChange={(e) =>
              setDraft((d) => ({ ...d, require_certified: e.target.checked }))
            }
            data-testid="require-certified-toggle"
            className="mt-0.5"
          />
          <span className="space-y-1">
            <span className="block font-medium">Require certified review</span>
            <span className="block text-xs text-muted-foreground">
              Translation sidecars are still emitted with{" "}
              <code className="rounded bg-muted px-1">certified=false</code> until a strong-auth
              custody event lands; this flag only steers the review queue.
            </span>
          </span>
        </label>
      </fieldset>

      <fieldset className="space-y-2">
        <legend className="text-sm font-semibold">Default quality tier</legend>
        <select
          value={draft.default_quality_tier}
          onChange={(e) =>
            setDraft((d) => ({
              ...d,
              default_quality_tier: e.target.value as TenantQualityTier,
            }))
          }
          data-testid="quality-tier-select"
          className="h-10 rounded-md border border-input bg-background px-3 text-sm"
        >
          {QUALITY_TIERS.map((tier) => (
            <option key={tier} value={tier}>
              {tier}
            </option>
          ))}
        </select>
        {errors.default_quality_tier ? (
          <p className="text-xs text-destructive" role="alert">
            {errors.default_quality_tier}
          </p>
        ) : null}
      </fieldset>

      <div className="flex items-center gap-2">
        <Button
          type="submit"
          disabled={saving || !dirty}
          data-testid="tenant-config-save"
        >
          {saving ? "Saving…" : "Save"}
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={handleDiscard}
          disabled={saving || !dirty}
          data-testid="tenant-config-discard"
        >
          Discard
        </Button>
        {dirty ? (
          <span className="text-xs text-muted-foreground" data-testid="tenant-config-dirty">
            Unsaved changes
          </span>
        ) : null}
      </div>
    </form>
  );
}

function toDraft(c: TenantConfig): TenantConfigUpdate {
  return {
    target_languages: [...(c.target_languages ?? [])],
    preferred_engines: [...(c.preferred_engines ?? [])],
    allow_nc_licensed: !!c.allow_nc_licensed,
    require_certified: !!c.require_certified,
    default_quality_tier: c.default_quality_tier ?? "standard",
  };
}
