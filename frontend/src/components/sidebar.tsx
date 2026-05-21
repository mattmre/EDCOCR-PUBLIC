"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";

interface NavItem {
  href: string;
  label: string;
  description: string;
}

const NAV_ITEMS: NavItem[] = [
  { href: "/dashboard", label: "Dashboard", description: "Pipeline summary" },
  { href: "/jobs", label: "Jobs", description: "Submitted documents" },
  { href: "/batches", label: "Batches", description: "Grouped OCR runs" },
  { href: "/review", label: "Review", description: "Human review queue" },
  { href: "/audit", label: "Audit", description: "Custody timeline" },
  { href: "/fleet", label: "Fleet", description: "Workers & queues" },
  { href: "/admin/tenants", label: "Tenants", description: "Tenants & glossary" },
  { href: "/admin/alerts", label: "Alerts", description: "Rules & channels" },
  { href: "/admin/features", label: "Feature Flags", description: "⚑ Toggles & change requests" },
  { href: "/settings", label: "Settings", description: "Browser preferences" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-60 shrink-0 border-r border-border bg-muted/30 md:flex md:flex-col">
      <div className="px-4 py-5">
        <p className="text-sm font-semibold">EDCOCR</p>
        <p className="text-xs text-muted-foreground">Operator Console</p>
      </div>
      <nav className="flex-1 space-y-1 px-2 pb-4" aria-label="Primary">
        {NAV_ITEMS.map((item) => {
          const active = pathname?.startsWith(item.href) ?? false;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "block rounded-md px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary text-primary-foreground"
                  : "text-foreground hover:bg-accent hover:text-accent-foreground"
              )}
              aria-current={active ? "page" : undefined}
            >
              <span className="block">{item.label}</span>
              <span
                className={cn(
                  "block text-xs",
                  active ? "text-primary-foreground/80" : "text-muted-foreground"
                )}
              >
                {item.description}
              </span>
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
        Operator Console
      </div>
    </aside>
  );
}
