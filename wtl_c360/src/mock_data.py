"""Generates a SMALL messy stand-in for the Maple dataset so the pipeline can be developed/tested
without the 4.6 GB download. Shape mimics the brief: 5 source systems, each with its own customer key,
CRM duplicates, mixed date formats, missing values, typos, household look-alikes, bad DPD values.
NOT the real column names - map real ones in config/schema_map.py after reading the release README."""
import random, hashlib, pandas as pd, numpy as np
from faker import Faker
from pathlib import Path

SNAP = pd.Timestamp("2026-09-28")
def make(out="data/raw", n=3000, seed=7):
    rnd, fk = random.Random(seed), Faker("en_CA"); Faker.seed(seed); np.random.seed(seed)
    Path(out).mkdir(parents=True, exist_ok=True)
    people = []
    for i in range(n):
        first, last = fk.first_name(), fk.last_name()
        people.append(dict(pid=i, first=first, last=last, dob=fk.date_of_birth(minimum_age=19, maximum_age=80),
            email=f"{first}.{last}{rnd.randint(1,99)}@{rnd.choice(['mail.ca','web.com','post.ca'])}".lower(),
            phone=f"{rnd.randint(200,989)}{rnd.randint(200,999)}{rnd.randint(1000,9999)}",
            postal=f"{rnd.choice('KLMNPV')}{rnd.randint(0,9)}{rnd.choice('ABCEGHJK')}{rnd.randint(0,9)}{rnd.choice('LMNPRST')}{rnd.randint(0,9)}",
            gender=rnd.choice(["F","M","X"]), marital=rnd.choice(["single","married","divorced"])))
    # household look-alikes: share last name, phone, postal but are different people (precision test)
    for i in rnd.sample(range(n), int(n*0.05)):
        p = dict(people[i]); p.update(pid=len(people), first=fk.first_name(), dob=fk.date_of_birth(minimum_age=19, maximum_age=80),
                  email=f"{p['last']}{rnd.randint(100,999)}@mail.ca".lower()); people.append(p)
    P = pd.DataFrame(people); N = len(P)
    nid = lambda pid: hashlib.sha256(f"sin{pid}".encode()).hexdigest()[:16]
    def fmt_dob(d):
        r = rnd.random(); d = pd.Timestamp(d)
        if r < .05: return None
        if r < .65: return d.strftime("%Y-%m-%d")
        if r < .85: return d.strftime("%d/%m/%Y")
        if r < .95: return d.strftime("%d %b %Y")
        return "00/00/0000" if r > .985 else d.strftime("%Y%m%d")
    def typo(s):
        if not s or rnd.random() > .10: return s
        k = rnd.randrange(len(s)); return s[:k] + s[k+1:] if rnd.random() < .5 else s[:k] + rnd.choice("aeiou") + s[k:]
    def fmt_phone(p):
        r = rnd.random()
        if r < .05: return None
        if r < .07: return "12345"
        return rnd.choice([p, f"({p[:3]}) {p[3:6]}-{p[6:]}", f"{p[:3]}-{p[3:6]}-{p[6:]}", f"+1{p}"])
    def fmt_postal(z):
        r = rnd.random()
        if r < .05: return None
        return z if r < .5 else (z[:3] + " " + z[3:]).lower() if r < .9 else "ZZZ999"
    def name_fmt(f, l): return rnd.choice([f"{f} {l}", f"{l}, {f}", f"{f} {l}".upper()])

    crm = []
    for r in P.itertuples():
        crm.append(dict(crm_customer_id=f"C{r.pid:07d}", first_name=r.first, last_name=r.last, dob=fmt_dob(r.dob),
            email=r.email if rnd.random() > .08 else None, phone=fmt_phone(r.phone), postal_code=fmt_postal(r.postal),
            national_id_hash=nid(r.pid) if rnd.random() > .05 else None, language=rnd.choice(["EN","EN","FR"]),
            segment=rnd.choice(["mass","affluent","student"]), hardship_flag=rnd.random() < .06,
            consent_sms=rnd.random() < .7, consent_email=rnd.random() < .75, consent_phone=rnd.random() < .9,
            gender=r.gender, marital_status=r.marital, truth_pid=r.pid))
    dupes = rnd.sample(range(N), int(N*.02)); k = N
    for i in dupes:
        d = dict(crm[i]); d["crm_customer_id"] = f"C{k:07d}"; k += 1
        d["first_name"] = typo(d["first_name"]); d["phone"] = fmt_phone(P.phone[i])
        d["national_id_hash"] = None if rnd.random() < .5 else d["national_id_hash"]
        crm.append(d)
    crm = pd.DataFrame(crm)
    crm.drop(columns="truth_pid").to_csv(f"{out}/crm_customers.csv", index=False)

    truth = [("crm", r.crm_customer_id, r.truth_pid) for r in crm.itertuples()]
    def system(name, prob, has_nid, has_email):
        rows = []
        for r in P.itertuples():
            if rnd.random() > prob: continue
            ref = f"{name[:2].upper()}{rnd.randint(10**7,10**8-1)}"; truth.append((name, ref, r.pid))
            rows.append(dict(src_customer_ref=ref, holder_name=name_fmt(typo(r.first), r.last), dob=fmt_dob(r.dob),
                phone=fmt_phone(r.phone), postal_code=fmt_postal(r.postal),
                email=r.email if has_email and rnd.random() < .5 else None,
                national_id_hash=nid(r.pid) if has_nid and rnd.random() < .6 else None))
        return pd.DataFrame(rows)
    sys_ = {"card": system("card", .7, False, False), "loan": system("loan", .5, False, True),
            "deposit": system("deposit", .6, False, False), "bureau": system("bureau", .55, True, False)}
    for nm, df in sys_.items():
        df = df.rename(columns={"src_customer_ref": "bank_subject_ref"}) if nm == "bureau" else df
        df.to_csv(f"{out}/{nm}_customers.csv", index=False)
    pd.DataFrame(truth, columns=["system","source_id","pid"]).to_csv(f"{out}/_truth_all.csv", index=False)

    def accounts(nm, prod):
        rows = []
        for r in sys_[nm].itertuples():
            od = rnd.random() < .18; dpd = rnd.choice([5,12,20,35,48,65,95]) if od else 0
            bal = round(rnd.uniform(200, 30000), 2); ov = round(bal*rnd.uniform(.02,.2),2) if od else 0
            if rnd.random() < .005: dpd = -rnd.randint(1,9)
            if rnd.random() < .004: ov = round(bal*1.5, 2)
            rows.append(dict(account_id=f"{prod[:2].upper()}A{rnd.randint(10**8,10**9-1)}", src_customer_ref=r.src_customer_ref,
                product_type=prod, balance=bal, overdue_amount=ov, days_past_due=dpd, snapshot_date="2026-09-28"))
        a = pd.DataFrame(rows); return pd.concat([a, a.sample(frac=.01, random_state=1)])
    for nm in ["card","loan","deposit"]: accounts(nm, nm).to_csv(f"{out}/{nm}_accounts.csv", index=False)
    bur = sys_["bureau"][["src_customer_ref"]].rename(columns={"src_customer_ref": "bank_subject_ref"})
    bur["bureau_score"] = [rnd.randint(300, 900) if rnd.random() > .04 else None for _ in range(len(bur))]
    bur.to_csv(f"{out}/bureau_scores.csv", index=False)

    cases, ptps, contacts, notes = [], [], [], []
    reasons = ["lost my job last month and cannot make the payment", "medical emergency in the family, need more time",
               "salary delayed by employer, will pay next week", "forgot about the payment, will pay today",
               "dispute the interest charged, wants a review", "hours cut at work, struggling with bills",
               "says the loan is paid already, confused", "promised to pay on Friday and sounded cooperative"]
    agents = [f"A{i:02d}" for i in range(1, 9)]
    sel = crm[crm.hardship_flag | (np.random.rand(len(crm)) < .2)]
    for r in sel.itertuples():
        cid = f"CS{len(cases):07d}"; ag = rnd.choice(agents)
        cases.append(dict(case_id=cid, crm_customer_id=r.crm_customer_id, status=rnd.choice(["open","open","open","on_hold","disputed","closed"]),
            stage=rnd.choice(["early","mid","late"]), agent_id=ag, opened_date=(SNAP - pd.Timedelta(days=rnd.randint(5,120))).strftime("%Y-%m-%d"),
            product_type=rnd.choice(["card","loan","deposit"])))
        for _ in range(rnd.randint(0, 3)):
            d_ = SNAP - pd.Timedelta(days=rnd.randint(0, 100))
            ptps.append(dict(ptp_id=f"PT{len(ptps):07d}", case_id=cid, crm_customer_id=r.crm_customer_id, promised_date=d_.strftime("%Y-%m-%d"),
                promised_amount=round(rnd.uniform(50, 800),2), status=rnd.choice(["KEPT","KEPT","BROKEN","BROKEN"])))
        for _ in range(rnd.randint(1, 4)):
            nid_ = f"N{len(notes):07d}"; txt = rnd.choice(reasons)
            if rnd.random() < .3: txt += f" Customer phone {r.phone} email {r.email}"
            notes.append(dict(note_id=nid_, crm_customer_id=r.crm_customer_id, note_text=txt))
            contacts.append(dict(contact_id=f"CT{len(contacts):07d}", case_id=cid, crm_customer_id=r.crm_customer_id,
                channel=rnd.choice(["phone","sms","email"]), contact_ts=(SNAP - pd.Timedelta(days=rnd.randint(0,60))).strftime("%Y-%m-%d"),
                outcome=rnd.choice(["answered","no_answer","promise_made","refused","voicemail"]), agent_id=ag, note_id=nid_))
    for nm, rows in [("collections_cases",cases),("promises_to_pay",ptps),("contact_history",contacts),("agent_notes",notes)]:
        pd.DataFrame(rows).to_csv(f"{out}/{nm}.csv", index=False)
    print(f"mock data: {len(crm)} CRM rows for {N} people -> {out}")
if __name__ == "__main__": make()
