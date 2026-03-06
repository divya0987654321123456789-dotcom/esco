import os
import pymysql
import streamlit as st

# Minimal DB config (duplicated intentionally to avoid importing app UI)
DB_CFG = dict(
    host=os.getenv("MYSQL_HOST", "localhost"),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    # Default DB name uses the clean schema; env MYSQL_DB overrides.
    database=os.getenv("MYSQL_DB", "esco_v23_clean"),
    cursorclass=pymysql.cursors.DictCursor,
)
_DB_FALLBACK = os.getenv("MYSQL_DB_FALLBACK", "esco_v23_clean")

def _open_mysql_or_create():
    try:
        conn = pymysql.connect(**DB_CFG)
        return True, conn, DB_CFG, None
    except Exception as e:
        # Handle ghost tablespace errors by retrying with fallback DB.
        try:
            err_txt = str(e)
            if "1813" in err_txt or "Tablespace for table" in err_txt:
                if DB_CFG.get("database") != _DB_FALLBACK:
                    cfg = dict(DB_CFG)
                    cfg["database"] = _DB_FALLBACK
                    conn = pymysql.connect(**cfg)
                    return True, conn, cfg, None
        except Exception:
            pass
        st.error(f"MySQL connection failed: {e}")
        return False, None, DB_CFG, e

def _ensure_company_tables(conn):
    with conn.cursor() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS company_details (
              id INT AUTO_INCREMENT PRIMARY KEY,
              company_name VARCHAR(100) NOT NULL UNIQUE,
              website VARCHAR(255), address VARCHAR(255), start_date DATE,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB CHARSET=utf8mb4;
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS company_capabilities (
              id INT AUTO_INCREMENT PRIMARY KEY,
              company_name VARCHAR(100) NOT NULL,
              capability_title VARCHAR(255),
              capability_description TEXT,
              naics_codes VARCHAR(255),
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              INDEX(company_name)
            ) ENGINE=InnoDB CHARSET=utf8mb4;
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS company_preferences (
              id INT AUTO_INCREMENT PRIMARY KEY,
              company_name VARCHAR(100) NOT NULL UNIQUE,
              deal_breakers TEXT, deal_makers TEXT,
              federal BOOLEAN DEFAULT TRUE, state_local BOOLEAN DEFAULT TRUE,
              preferred_states VARCHAR(255), preferred_countries VARCHAR(255),
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB CHARSET=utf8mb4;
            """
        )
    conn.commit()

    # --- Lightweight schema migration for existing deployments ---
    try:
        with conn.cursor() as c:
            # Check existing columns in company_details
            c.execute(
                "SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME='company_details'",
                (DB_CFG.get("database"),),
            )
            cols = {row["COLUMN_NAME"].lower() for row in c.fetchall()}

            def _alter(sql: str):
                try:
                    c.execute(sql)
                except Exception:
                    pass

            # Add missing columns if the table pre-existed without them
            if "company_name" not in cols:
                _alter("ALTER TABLE company_details ADD COLUMN company_name VARCHAR(100) NULL")
                # If a legacy 'name' column exists, migrate values
                if "name" in cols:
                    _alter("UPDATE company_details SET company_name = name WHERE company_name IS NULL AND name IS NOT NULL")
                # Try to add unique index (ignore if fails due to duplicates/nulls)
                _alter("ALTER TABLE company_details ADD UNIQUE KEY uniq_company_name (company_name)")
            if "website" not in cols:
                _alter("ALTER TABLE company_details ADD COLUMN website VARCHAR(255) NULL")
            if "address" not in cols:
                _alter("ALTER TABLE company_details ADD COLUMN address VARCHAR(255) NULL")
            if "start_date" not in cols:
                _alter("ALTER TABLE company_details ADD COLUMN start_date DATE NULL")
        conn.commit()
    except Exception:
        # Best-effort migration; proceed even if checks fail
        pass

def upsert_company_details(conn, company_name, website, address, start_date):
    _ensure_company_tables(conn)
    with conn.cursor() as c:
        c.execute(
            (
                "INSERT INTO company_details (company_name, website, address, start_date) "
                "VALUES (%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE website=VALUES(website), address=VALUES(address), start_date=VALUES(start_date)"
            ),
            (company_name, website, address, start_date),
        )
    conn.commit()

def upsert_company_preferences(conn, company_name, deal_breakers, deal_makers, federal, state_local, preferred_states, preferred_countries):
    _ensure_company_tables(conn)
    with conn.cursor() as c:
        c.execute(
            (
                "INSERT INTO company_preferences (company_name, deal_breakers, deal_makers, federal, state_local, preferred_states, preferred_countries) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE deal_breakers=VALUES(deal_breakers), deal_makers=VALUES(deal_makers), federal=VALUES(federal), state_local=VALUES(state_local), preferred_states=VALUES(preferred_states), preferred_countries=VALUES(preferred_countries)"
            ),
            (company_name, deal_breakers, deal_makers, federal, state_local, preferred_states, preferred_countries),
        )
    conn.commit()

def add_company_capability(conn, company_name, title, desc, naics):
    _ensure_company_tables(conn)
    with conn.cursor() as c:
        c.execute(
            (
                "INSERT INTO company_capabilities (company_name, capability_title, capability_description, naics_codes) "
                "VALUES (%s,%s,%s,%s)"
            ),
            (company_name, title, desc, naics),
        )
    conn.commit()

__all__ = [
    "_open_mysql_or_create",
    "_ensure_company_tables",
    "upsert_company_details",
    "upsert_company_preferences",
    "add_company_capability",
]


