"""ONE place to change when you meet the real column names (read maple_collections_release/README.md +
DATA_DICTIONARY.xlsx, then edit this file). Everything downstream uses these logical names only."""
RAW_DIR = "data/raw"          # real data: "maple_data/maple_collections_release/data" (check README for the exact folder)
SNAPSHOT_DATE = "2026-09-28"  # given in the brief

# identity sources -> which column holds which logical identity field (None = not present in that system)
IDENTITY = {
  "crm":     dict(table="crm_customers",     id="crm_customer_id",  first="first_name", last="last_name", name=None,
                  dob="dob", email="email", phone="phone", postal="postal_code", nid="national_id_hash"),
  "card":    dict(table="card_customers",    id="src_customer_ref", first=None, last=None, name="holder_name",
                  dob="dob", email="email", phone="phone", postal="postal_code", nid="national_id_hash"),
  "loan":    dict(table="loan_customers",    id="src_customer_ref", first=None, last=None, name="holder_name",
                  dob="dob", email="email", phone="phone", postal="postal_code", nid="national_id_hash"),
  "deposit": dict(table="deposit_customers", id="src_customer_ref", first=None, last=None, name="holder_name",
                  dob="dob", email="email", phone="phone", postal="postal_code", nid="national_id_hash"),
  "bureau":  dict(table="bureau_customers",  id="bank_subject_ref", first=None, last=None, name="holder_name",
                  dob="dob", email="email", phone="phone", postal="postal_code", nid="national_id_hash"),
}
# product account tables -> (identity system whose id they carry, id column)
ACCOUNTS = {"card": ("card_accounts", "src_customer_ref"), "loan": ("loan_accounts", "src_customer_ref"),
            "deposit": ("deposit_accounts", "src_customer_ref")}
BUREAU_SCORES = ("bureau_scores", "bank_subject_ref", "bureau_score")
# match thresholds (precision first)
AUTO_SCORE, STEWARD_SCORE = 0.92, 0.75
# columns that must NEVER reach Layer 2 / the model (protected attributes)
PROTECTED = ["gender", "marital_status", "citizenship", "household", "newcomer", "accessibility", "vulnerability", "accent"]
