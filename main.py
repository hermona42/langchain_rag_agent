"""Application entry point."""

from config.settings import settings


def main() -> None:
    print(f"Starting {settings.app_name} in {settings.environment} mode")


if __name__ == "__main__":
    main()
