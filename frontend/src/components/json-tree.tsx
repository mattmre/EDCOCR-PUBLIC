"use client";

import { useState } from "react";
import { cn } from "@/lib/cn";

interface JsonTreeProps {
  value: unknown;
  /** Indentation level used internally; consumers should leave at 0. */
  depth?: number;
  /** Property key being rendered (for nested values). */
  name?: string;
}

const TOKEN_COLORS = {
  key: "text-foreground/80",
  string: "text-emerald-700",
  number: "text-amber-700",
  boolean: "text-violet-700",
  null: "text-muted-foreground",
  punct: "text-muted-foreground",
};

/**
 * Tiny zero-dep JSON pretty-printer with collapsible objects and arrays.
 *
 * Designed for custody event payloads which are small and shallow. The
 * component is fully synchronous; no syntax-highlight plugin needed.
 */
export function JsonTree({ value, depth = 0, name }: JsonTreeProps) {
  if (value === null) {
    return <Leaf name={name} text="null" colorClass={TOKEN_COLORS.null} depth={depth} />;
  }
  const t = typeof value;
  if (t === "string") {
    return (
      <Leaf
        name={name}
        text={JSON.stringify(value)}
        colorClass={TOKEN_COLORS.string}
        depth={depth}
      />
    );
  }
  if (t === "number" || t === "bigint") {
    return (
      <Leaf
        name={name}
        text={String(value)}
        colorClass={TOKEN_COLORS.number}
        depth={depth}
      />
    );
  }
  if (t === "boolean") {
    return (
      <Leaf
        name={name}
        text={value ? "true" : "false"}
        colorClass={TOKEN_COLORS.boolean}
        depth={depth}
      />
    );
  }
  if (Array.isArray(value)) {
    return <ArrayNode name={name} value={value} depth={depth} />;
  }
  if (t === "object") {
    return <ObjectNode name={name} value={value as Record<string, unknown>} depth={depth} />;
  }
  return (
    <Leaf
      name={name}
      text={String(value)}
      colorClass={TOKEN_COLORS.punct}
      depth={depth}
    />
  );
}

interface LeafProps {
  name?: string;
  text: string;
  colorClass: string;
  depth: number;
}

function Leaf({ name, text, colorClass }: LeafProps) {
  return (
    <div className="flex items-baseline gap-1 font-mono text-xs leading-5">
      {name !== undefined ? (
        <>
          <span className={TOKEN_COLORS.key}>{JSON.stringify(name)}</span>
          <span className={TOKEN_COLORS.punct}>:</span>
        </>
      ) : null}
      <span className={cn("break-all", colorClass)}>{text}</span>
    </div>
  );
}

interface ContainerProps {
  name?: string;
  depth: number;
}

function ObjectNode({
  name,
  value,
  depth,
}: ContainerProps & { value: Record<string, unknown> }) {
  const keys = Object.keys(value);
  const [open, setOpen] = useState(depth < 1);

  const summary = keys.length === 0 ? "{}" : `{${keys.length} ${keys.length === 1 ? "field" : "fields"}}`;

  return (
    <div className="font-mono text-xs leading-5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "inline-flex items-baseline gap-1 text-left",
          "hover:text-primary"
        )}
        aria-expanded={open}
      >
        <span className="w-3 select-none text-muted-foreground">{open ? "▾" : "▸"}</span>
        {name !== undefined ? (
          <>
            <span className={TOKEN_COLORS.key}>{JSON.stringify(name)}</span>
            <span className={TOKEN_COLORS.punct}>:</span>
          </>
        ) : null}
        <span className={TOKEN_COLORS.punct}>{open ? "{" : summary}</span>
      </button>
      {open ? (
        <div className="ml-4 border-l border-border/60 pl-3">
          {keys.map((k) => (
            <JsonTree key={k} name={k} value={value[k]} depth={depth + 1} />
          ))}
          <div className={TOKEN_COLORS.punct}>{"}"}</div>
        </div>
      ) : null}
    </div>
  );
}

function ArrayNode({
  name,
  value,
  depth,
}: ContainerProps & { value: unknown[] }) {
  const [open, setOpen] = useState(depth < 1);
  const summary = value.length === 0 ? "[]" : `[${value.length}]`;
  return (
    <div className="font-mono text-xs leading-5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-baseline gap-1 text-left hover:text-primary"
        aria-expanded={open}
      >
        <span className="w-3 select-none text-muted-foreground">{open ? "▾" : "▸"}</span>
        {name !== undefined ? (
          <>
            <span className={TOKEN_COLORS.key}>{JSON.stringify(name)}</span>
            <span className={TOKEN_COLORS.punct}>:</span>
          </>
        ) : null}
        <span className={TOKEN_COLORS.punct}>{open ? "[" : summary}</span>
      </button>
      {open ? (
        <div className="ml-4 border-l border-border/60 pl-3">
          {value.map((item, idx) => (
            <JsonTree key={idx} value={item} depth={depth + 1} />
          ))}
          <div className={TOKEN_COLORS.punct}>{"]"}</div>
        </div>
      ) : null}
    </div>
  );
}
