import argparse
import getpass
import json
import os

from app.core.passwords import hash_password


def _parsed_roles(raw: str) -> list[str]:
    roles = [part.strip() for part in str(raw).split(",") if part.strip()]
    return roles or ["user"]


def _resolve_password(args) -> str:
    if args.password:
        return str(args.password)
    prompt_user = str(args.username).strip() or "user"
    return getpass.getpass(f"Password for {prompt_user}: ").strip()


def _load_existing_users() -> list[dict]:
    raw = os.getenv("SECURITY_LOGIN_USERS_JSON", "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create or update an auth user entry for SECURITY_LOGIN_USERS_JSON "
            "(Postgres auth was removed)."
        )
    )
    parser.add_argument("--username", required=True, help="Login username.")
    parser.add_argument(
        "--user-id",
        default="",
        help="JWT subject/user id. Defaults to username.",
    )
    parser.add_argument(
        "--password", default="", help="Plaintext password (or leave empty for prompt)."
    )
    parser.add_argument(
        "--roles", default="admin", help='Comma-separated roles. Example: "admin,user"'
    )
    parser.add_argument(
        "--inactive",
        action="store_true",
        help="Mark user inactive (blocked login).",
    )
    args = parser.parse_args()

    username = str(args.username).strip()
    if not username:
        raise RuntimeError("username is required.")
    user_id = str(args.user_id).strip() or username
    password = _resolve_password(args)
    if not password:
        raise RuntimeError("password is required.")

    users = _load_existing_users()
    users_by_name = {
        str(item.get("username", "")).strip().lower(): item
        for item in users
        if str(item.get("username", "")).strip()
    }
    users_by_name[username.lower()] = {
        "username": username,
        "user_id": user_id,
        "password_hash": hash_password(password),
        "roles": _parsed_roles(args.roles),
        "is_active": not bool(args.inactive),
    }

    merged = sorted(users_by_name.values(), key=lambda item: str(item.get("username", "")).lower())
    serialized = json.dumps(merged, separators=(",", ":"))
    print("Updated SECURITY_LOGIN_USERS_JSON value:")
    print(serialized)


if __name__ == "__main__":
    main()
