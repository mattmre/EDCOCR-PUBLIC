/**
 * Tiny class-name joiner. We keep this in-tree instead of pulling in `clsx`
 * to keep the dependency surface small for Wave 1.
 */
export function cn(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}
