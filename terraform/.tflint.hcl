# TFLint configuration for OCR-Local Terraform modules
#
# Uses only the built-in terraform plugin to keep CI fast.
# Cloud-specific rule plugins (aws, google, oci) are omitted
# because they require provider schema downloads and add
# significant CI time for marginal first-run value.

plugin "terraform" {
  enabled = true
  version = "0.9.1"
  source  = "github.com/terraform-linters/tflint-ruleset-terraform"
}

# -----------------------------------------------------------------------
# Rule overrides
# -----------------------------------------------------------------------

# Warn on deprecated syntax
rule "terraform_deprecated_interpolation" {
  enabled = true
}

# Require documented variables and outputs
rule "terraform_documented_variables" {
  enabled = true
}

rule "terraform_documented_outputs" {
  enabled = true
}

# Enforce consistent naming conventions
rule "terraform_naming_convention" {
  enabled = true
}

# Flag unused declarations
rule "terraform_unused_declarations" {
  enabled = true
}

# The following rules are disabled because they flag pre-existing structural
# issues in the shared module and environment configs that require a larger
# refactor (follow-on task, not N1-A scope):
# - standard_module_structure: shared module uses per-concern files
# - unused_required_providers: env configs declare providers for all clouds
# - required_providers: env configs inherit constraints from child modules
rule "terraform_standard_module_structure" {
  enabled = false
}

rule "terraform_unused_required_providers" {
  enabled = false
}

rule "terraform_required_providers" {
  enabled = false
}
