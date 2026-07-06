import duckdb
con = duckdb.connect('data/churn.duckdb')
df = con.execute("""
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_name IN ('transactions','user_logs','cohorts_s')
""").df()
print(df)
