"""
Data Quality Validation and Schema Enforcement Module.
Applies quality contracts, value constraints, range bounds, and referential integrity checks.
"""

import os
import yaml
import pandas as pd
from typing import Dict, Any, Tuple, List


class DataValidator:
    """Validates ingested datasets against declared data quality contracts."""

    def __init__(self, rules_path: str = None):
        if rules_path is None:
            rules_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "quality_rules.yaml"
            )
        self.rules = self._load_rules(rules_path)
        self.report = {}

    def _load_rules(self, rules_path: str) -> Dict[str, Any]:
        if os.path.exists(rules_path):
            with open(rules_path, "r") as f:
                cfg = yaml.safe_load(f)
                return cfg.get("quality_rules", {})
        return {}

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate dataframe against rules for dataset_name. Returns (clean_df, metrics)."""
        rules = self.rules.get(dataset_name, {})
        initial_count = len(df)
        violations = []
        valid_mask = pd.Series(True, index=df.index)

        # 1. Required Columns Check
        required_cols = rules.get("required_columns", [])
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"[{dataset_name}] Missing critical columns: {missing_cols}")

        # 2. Not-Null Checks
        not_null_cols = rules.get("not_null", [])
        for col in not_null_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    violations.append(f"Found {null_count} nulls in required column '{col}'")
                    valid_mask &= df[col].notnull()

        # 3. Numeric Range Checks
        ranges = rules.get("ranges", {})
        for col, bounds in ranges.items():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                min_val = bounds.get("min")
                max_val = bounds.get("max")
                if min_val is not None:
                    out_of_bounds = (s < min_val).sum()
                    if out_of_bounds > 0:
                        violations.append(f"Column '{col}' had {out_of_bounds} values below min {min_val}")
                        valid_mask &= (s >= min_val) | s.isnull()
                if max_val is not None:
                    out_of_bounds = (s > max_val).sum()
                    if out_of_bounds > 0:
                        violations.append(f"Column '{col}' had {out_of_bounds} values above max {max_val}")
                        valid_mask &= (s <= max_val) | s.isnull()

        # 4. Categorical Allowed Values
        allowed = rules.get("allowed_values", {})
        for col, valid_vals in allowed.items():
            if col in df.columns:
                invalid_count = (~df[col].isin(valid_vals) & df[col].notnull()).sum()
                if invalid_count > 0:
                    violations.append(f"Column '{col}' had {invalid_count} unapproved categorical values")
                    valid_mask &= df[col].isin(valid_vals) | df[col].isnull()

        clean_df = df[valid_mask].copy()
        dropped_count = initial_count - len(clean_df)

        metrics = {
            "initial_rows": initial_count,
            "passed_rows": len(clean_df),
            "quarantined_rows": dropped_count,
            "passed": (dropped_count == 0),
            "violations": violations
        }
        self.report[dataset_name] = metrics
        
        status_msg = "PASSED" if metrics["passed"] else f"WARNING ({dropped_count} quarantined)"
        print(f"[Data Quality] {dataset_name}: {status_msg} ({metrics['passed_rows']}/{metrics['initial_rows']} rows)")
        return clean_df, metrics

    def validate_referential_integrity(
        self,
        contracts_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        tickets_df: pd.DataFrame,
        usage_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Verify customer_id foreign key consistency across all 3 source systems."""
        master_ids = set(contracts_df["customer_id"].unique())
        
        orphan_invoices = set(invoices_df["customer_id"].unique()) - master_ids
        orphan_tickets = set(tickets_df["customer_id"].unique()) - master_ids
        orphan_usage = set(usage_df["customer_id"].unique()) - master_ids
        
        results = {
            "master_customers": len(master_ids),
            "orphan_invoice_customers": len(orphan_invoices),
            "orphan_ticket_customers": len(orphan_tickets),
            "orphan_usage_customers": len(orphan_usage),
            "referential_integrity_intact": (len(orphan_invoices) == 0 and len(orphan_tickets) == 0 and len(orphan_usage) == 0)
        }
        self.report["referential_integrity"] = results
        print(f"[Data Quality] Referential Integrity Check: {'PASSED' if results['referential_integrity_intact'] else 'FAILED'}")
        return results
