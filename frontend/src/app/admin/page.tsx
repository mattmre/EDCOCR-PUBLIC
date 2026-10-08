import { redirect } from "next/navigation";

// FE001: bare /admin has no index. No nav link points here (sidebar links
// only to /admin/tenants, /admin/alerts, /admin/features), so redirect
// direct-URL visits to the primary admin surface instead of 404ing.
export default function AdminIndexPage() {
  redirect("/admin/tenants");
}
