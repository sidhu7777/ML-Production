import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine, text

eng = create_engine(os.getenv("DATABASE_URL"), connect_args={"connect_timeout": 20})
with eng.connect() as c:
    print("-- thresholds rows with is_default flag --")
    rows = c.execute(text("SELECT id, user_id, is_default, rsrp_json FROM thresholds ORDER BY id")).fetchall()
    for r in rows:
        print(dict(r._mapping))
    print()

    print("-- tbl_user columns --")
    cols = c.execute(text("SHOW COLUMNS FROM tbl_user")).fetchall()
    print([c2[0] for c2 in cols])
    print()

    print("-- users in company_id=23 --")
    users = c.execute(text("SELECT id, company_id, email FROM tbl_user WHERE company_id=23 LIMIT 10")).fetchall()
    for u in users:
        print(dict(u._mapping))
