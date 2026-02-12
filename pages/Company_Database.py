import streamlit as st
import pymysql
from datetime import date
from db_helpers import (
    _open_mysql_or_create,
    _ensure_company_tables,
    upsert_company_details,
    upsert_company_preferences,
    add_company_capability,
)

st.set_page_config(page_title="Company Database", layout="wide")
st.title("🏢 Company Database Manager")

ok, conn, _, _ = _open_mysql_or_create()
if not ok:
    st.error("Database connection failed.")
    st.stop()
_ensure_company_tables(conn)

companies = ["IKIO", "IKIO ENERGY Engineering", "Ikio Energy"]
company = st.selectbox("Select Company", companies)

tab1, tab2, tab3 = st.tabs(["Company Details", "Capabilities", "Preferences"])

# ---------- DETAILS ----------
with tab1:
    st.subheader("Company Details")
    name = st.text_input("Business Name", company)
    address = st.text_input("Business Address", "")
    website = st.text_input("Website", "")
    start_date = st.date_input("Start Date", date.today())
    if st.button("💾 Save Details"):
        upsert_company_details(conn, name, website, address, start_date)
        st.success("Details saved!")

# ---------- CAPABILITIES ----------
with tab2:
    st.subheader("Capabilities")
    cap_title = st.text_input("Capability Title")
    cap_desc = st.text_area("Capability Description")
    cap_naics = st.text_input("NAICS Codes")
    if st.button("➕ Add Capability"):
        add_company_capability(conn, company, cap_title, cap_desc, cap_naics)
        st.success("Capability added!")

# ---------- PREFERENCES ----------
with tab3:
    st.subheader("Preferences")
    deal_breakers = st.text_area("Deal Breakers")
    deal_makers = st.text_area("Deal Makers")
    federal = st.checkbox("Federal", True)
    state_local = st.checkbox("State & Local", True)
    states = st.text_input("Preferred States (comma-separated)", "Texas")
    countries = st.text_input("Preferred Countries", "United States")
    if st.button("💾 Save Preferences"):
        upsert_company_preferences(conn, company, deal_breakers, deal_makers, federal, state_local, states, countries)
        st.success("Preferences saved!")

st.info("Changes are saved instantly and used in the next RFP evaluation.")
