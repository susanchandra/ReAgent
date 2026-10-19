import os
import sys
from datetime import datetime, timezone
from openai import OpenAI

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
key = os.environ.get("NEBIUS_API_KEY", "").strip()

print("UTC timestamp:", datetime.now(timezone.utc).isoformat())
print("Endpoint:", BASE_URL)

if not key:
    print("RESULT: NEBIUS_API_KEY is not set in this PowerShell session.")
    sys.exit(2)

client = OpenAI(
    api_key=key,
    base_url=BASE_URL,
    timeout=15.0,
    max_retries=0,
)

try:
    response = client.models.list()
    print("HTTP result: SUCCESS")
    print("Models returned:", len(response.data))
    print("Model IDs:", ", ".join(model.id for model in response.data[:20]))
except Exception as exc:
    print("HTTP result: FAILED")
    print("Error type:", type(exc).__name__)
    print("HTTP status:", getattr(exc, "status_code", "unknown"))
    print("Request ID:", getattr(exc, "request_id", None) or "not exposed by SDK")
    # Do not print exception text: it can contain request details.
    sys.exit(1)
