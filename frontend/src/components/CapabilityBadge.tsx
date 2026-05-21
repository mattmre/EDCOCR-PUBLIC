import { cn } from "@/lib/cn";

const CAPABILITY_PALETTE: Record<string, string> = {
  ocr_gpu: "bg-blue-100 text-blue-800 border-blue-300",
  ocr_cpu: "bg-emerald-100 text-emerald-800 border-emerald-300",
  nlp: "bg-purple-100 text-purple-800 border-purple-300",
  layoutlm: "bg-amber-100 text-amber-800 border-amber-300",
  translation: "bg-pink-100 text-pink-800 border-pink-300",
  compression: "bg-slate-100 text-slate-800 border-slate-300",
};

const FALLBACK = "bg-gray-100 text-gray-800 border-gray-300";

interface CapabilityBadgeProps {
  capability: string;
  className?: string;
}

export function CapabilityBadge({ capability, className }: CapabilityBadgeProps) {
  const palette = CAPABILITY_PALETTE[capability] ?? FALLBACK;
  return (
    <span
      data-testid={`capability-badge-${capability}`}
      data-palette={palette}
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium",
        palette,
        className
      )}
    >
      {capability}
    </span>
  );
}
