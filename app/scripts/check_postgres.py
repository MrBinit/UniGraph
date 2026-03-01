from app.infra.postgres_client import close_postgres_pool, verify_postgres_connection


def main():
    """Verify the configured Postgres connection and print basic server details."""
    details = verify_postgres_connection()
    print("Postgres connection verified.")
    print(f"database={details.get('database_name', '')}")
    print(f"schema={details.get('schema_name', '')}")
    print(f"user={details.get('user_name', '')}")
    print(f"server={details.get('server_version', '')}")
    close_postgres_pool()


if __name__ == "__main__":
    main()
