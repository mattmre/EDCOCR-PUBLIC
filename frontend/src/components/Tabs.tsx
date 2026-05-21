"use client";

import { useState, type ReactNode } from "react";
import { cn } from "@/lib/cn";

export interface TabDefinition {
  id: string;
  label: string;
  content: ReactNode;
}

export interface TabsProps {
  tabs: TabDefinition[];
  defaultTab?: string;
  onChange?: (tabId: string) => void;
  className?: string;
}

export function Tabs({ tabs, defaultTab, onChange, className }: TabsProps) {
  const initial = defaultTab && tabs.some((t) => t.id === defaultTab) ? defaultTab : tabs[0]?.id;
  const [active, setActive] = useState<string | undefined>(initial);
  const activeTab = tabs.find((t) => t.id === active) ?? tabs[0];

  function handleSelect(id: string) {
    setActive(id);
    onChange?.(id);
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div role="tablist" className="flex gap-1 border-b border-border">
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab?.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              data-testid={`tab-${tab.id}`}
              className={cn(
                "px-3 py-2 text-sm font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                isActive
                  ? "border-b-2 border-primary text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
              onClick={() => handleSelect(tab.id)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div role="tabpanel" data-testid={`tabpanel-${activeTab?.id ?? "none"}`}>
        {activeTab?.content}
      </div>
    </div>
  );
}
