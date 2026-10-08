# Optional ArgoCD overlays

The federation `ApplicationSet` may reference `values-<region>.yaml` and
`values-<tenantClass>.yaml` files under this directory. They are optional:
ArgoCD is configured with `ignoreMissingValueFiles: true`, so a cluster can
use the chart defaults until an operator adds a reviewed overlay.

Overlay names are data supplied by the ApplicationSet generator; do not add
untrusted path components to generator entries.
