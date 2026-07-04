"""
One-off re-authorization for Google Drive access, using CommandLineAuth
instead of LocalWebserverAuth - no port forwarding or local webserver
needed. Just visit the printed URL on any device, approve access, and
paste the verification code back here.

Run this once; it saves fresh credentials to secrets/pydrive_credentials.json.
After this succeeds, download_intermediate_assets.py should run normally.
"""
import sys
sys.path.insert(0, '/scratch/s1214882/gaza-damage-mapping')

from pydrive2.auth import GoogleAuth
from src.constants import SECRETS_PATH

GoogleAuth.DEFAULT_SETTINGS["client_config_file"] = str(SECRETS_PATH / "client_secrets.json")
gauth = GoogleAuth(settings_file=str(SECRETS_PATH / "pydrive_settings.yaml"))

print("Starting command-line authorization...")
print("A URL will appear below. Open it on any device's browser, sign in,")
print("approve access, then copy the verification code shown and paste it here.")
print()

gauth.CommandLineAuth()

print()
print("Authorization successful. Credentials saved to:")
print(f"  {SECRETS_PATH / 'pydrive_credentials.json'}")
