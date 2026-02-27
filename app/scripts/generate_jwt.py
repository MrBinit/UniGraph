import argparse

from app.core.security import create_access_token


def main():
    parser = argparse.ArgumentParser(description="Generate a JWT for local testing.")
    parser.add_argument("--user-id", required=True, help="Token subject/user_id")
    parser.add_argument(
        "--roles",
        default="user",
        help="Comma-separated roles, e.g. user,admin",
    )
    parser.add_argument(
        "--expires-minutes",
        type=int,
        default=None,
        help="Optional token expiration override in minutes.",
    )
    args = parser.parse_args()

    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    token = create_access_token(
        user_id=args.user_id,
        roles=roles or ["user"],
        expires_minutes=args.expires_minutes,
    )
    print(token)


if __name__ == "__main__":
    main()
