#!/usr/bin/env python3
"""
Re-authorise Google Drive access for the pipeline (PyDrive2) on a headless server.

    python3 reauth_gdrive.py            # sign in again, save new credentials, then test them
    python3 reauth_gdrive.py --check    # only test the saved credentials (no sign-in)
    python3 reauth_gdrive.py --local    # run on a computer WITH a browser (laptop): no copy/paste

Why this replaces the old version
  * PyDrive2 always tries to REFRESH a saved token first and only asks for a login when
    there is no refresh token at all. With a dead refresh token, CommandLineAuth() stops
    with "invalid_grant" before it prints any link. This script ignores the saved token.
  * CommandLineAuth() also used Google's "out-of-band" redirect (urn:ietf:wg:oauth:2.0:oob),
    which Google has been shutting down since 2022. Desktop-app clients use a loopback
    redirect (http://localhost:PORT) instead. This script does that without port
    forwarding: you open the link on any device, approve, and paste back the address
    the browser ends up on.

New credentials are written to the file named by save_credentials_file in
secrets/pydrive_settings.yaml (normally secrets/pydrive_credentials.json). Nothing is
changed unless the sign-in succeeds. No tokens are ever printed.
"""
import argparse
import http.server
import os
import re
import sys
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from pydrive2.auth import AuthenticationError, GoogleAuth, RefreshError
from pydrive2.drive import GoogleDrive

# Loopback address. Nothing has to be listening on it: after you approve, the browser is
# sent there, and the address bar then holds the one-time code that you paste back here.
REDIRECT_URI = "http://localhost:8080/"
# --local mode: this script itself listens here and catches the redirect automatically.
LOCAL_REDIRECT_URI = "http://127.0.0.1:8080/"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def find_repo_root():
    here = Path(__file__).resolve().parent
    for start in (here, Path.cwd().resolve()):
        for folder in (start, *start.parents):
            if (folder / "secrets" / "pydrive_settings.yaml").is_file():
                return folder
    sys.exit(
        "Could not find secrets/pydrive_settings.yaml. Keep this script inside the "
        "repository (next to the secrets/ folder) and run it again."
    )


def extract_code(pasted, expected_state):
    """Authorisation code from what was pasted: the full address, its query string, or the bare code."""
    text = _ANSI.sub("", pasted).strip().strip("'\"<>").strip()
    if not text:
        raise ValueError("nothing was pasted.")
    if "code=" in text or "error=" in text or "://" in text or "?" in text:
        query = text.split("?", 1)[1] if "?" in text else text
        query = query.split("#", 1)[0]
        params = {k: v[0] for k, v in parse_qs(query).items()}
        if "error" in params:
            detail = params.get("error_description")
            raise ValueError(
                f"Google returned an error instead of a code: {params['error']}"
                + (f" ({detail})" if detail else "")
                + ". If you pressed Cancel/Deny, run the script again and press Allow."
            )
        code = params.get("code")
        if not code:
            raise ValueError(
                "the pasted text has no 'code=' part. After you approve, the browser lands on an "
                "address starting http://localhost:8080/?state=...&code=...; copy that whole address."
            )
        state = params.get("state")
        if state is not None and state != expected_state:
            raise ValueError(
                "this address belongs to a different sign-in attempt (the 'state' does not match). "
                "Run the script again and use the link it prints this time."
            )
        return code
    return unquote(text)  # a bare code, possibly still percent-encoded (4%2F0A... -> 4/0A...)


def _gb(n):
    return f"{int(n) / 2**30:.1f} GB"


def start_catcher():
    """Start a one-shot local web server that records the ?code=... Google redirects the browser to."""
    box = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if "code" in query or "error" in query:
                box.update(query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body style='font-family:sans-serif'><h3>Done. You can close this "
                             b"tab and go back to the terminal.</h3></body></html>")

        def log_message(self, *args):
            pass

    return http.server.HTTPServer(("127.0.0.1", 8080), Handler), box


def wait_for_redirect(server, box, timeout=300):
    server.timeout = 1
    deadline = time.time() + timeout
    while not box and time.time() < deadline:
        server.handle_request()
    server.server_close()
    return dict(box)


def sign_in(settings_file, secrets_dir, local=False):
    gauth = GoogleAuth(settings_file=str(settings_file))
    # The old script set this on GoogleAuth.DEFAULT_SETTINGS, which is ignored when the YAML
    # loads; fall back to secrets/client_secrets.json only if the configured file is missing.
    if gauth.settings.get("client_config_backend") == "file" and not Path(
        gauth.settings.get("client_config_file") or ""
    ).is_file():
        gauth.settings["client_config_file"] = str(secrets_dir / "client_secrets.json")
    gauth.GetFlow()

    flow = gauth.flow
    flow.redirect_uri = LOCAL_REDIRECT_URI if local else REDIRECT_URI
    flow.params.pop("approval_prompt", None)  # deprecated, and cannot be combined with prompt
    flow.params.update({"access_type": "offline", "prompt": "consent"})  # always issue a refresh token
    state = uuid.uuid4().hex
    url = flow.step1_get_authorize_url(state=state)
    target = Path(gauth.settings.get("save_credentials_file") or secrets_dir / "pydrive_credentials.json")
    if not target.parent.is_dir():  # e.g. an absolute Linux path while running on a laptop
        target = secrets_dir / "pydrive_credentials.json"
    server = box = None
    if local:
        try:
            server, box = start_catcher()
        except OSError as err:
            print(f"Could not listen on 127.0.0.1:8080 ({err}). Close whatever uses port 8080 and try again.")
            return 1

    print("Re-authorising Google Drive for the pipeline")
    print(f"  OAuth client : {gauth.client_config.get('client_id')}")
    client_file = gauth.settings.get("client_config_file")
    print(f"  Client file  : {Path(client_file).resolve() if client_file else '(from pydrive_settings.yaml)'}")
    print(f"  Will save to : {target.resolve()}")
    print()
    if local:
        print("Opening your browser. If it does not open, copy this link into a browser on THIS computer")
        print("and sign in with the Google account whose Drive holds the Earth Engine exports:")
        print()
        print(url)
        print()
        print('If Google says it "hasn\'t verified this app", click Advanced, then "Go to ... (unsafe)",')
        print("then tick the Drive box if shown and click Continue / Allow. This window finishes by itself.")
        print()
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            got = wait_for_redirect(server, box)
        except KeyboardInterrupt:
            print("\nCancelled. Nothing was changed.")
            return 1
        if not got:
            print("Timed out after 5 minutes without hearing back from the browser. Nothing was changed.")
            return 1
        pasted = LOCAL_REDIRECT_URI + "?" + urlencode(got)
    else:
        print("STEP 1  Open this link in a browser (any device) and sign in with the Google account")
        print("        whose Drive holds the Earth Engine exports:")
        print()
        print(url)
        print()
        print('STEP 2  Approve access. If Google says it "hasn\'t verified this app", click Advanced,')
        print('        then "Go to ... (unsafe)", then Continue / Allow.')
        print()
        print("STEP 3  Your browser then lands on a page that does NOT load (its address starts with")
        print("        http://localhost:8080/?state=...). That is expected. Copy the WHOLE address from")
        print("        the address bar and paste it below.")
        print()
        try:
            pasted = input("Paste the address (or just the code) here, then press Enter: ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled. Nothing was changed.")
            return 1
    try:
        code = extract_code(pasted, state)
    except ValueError as err:
        print(f"\nCould not use what was pasted: {err}\nNothing was changed.")
        return 1
    try:
        gauth.Authenticate(code)
    except AuthenticationError as err:
        print(f"\nGoogle rejected the code: {err}")
        print("Codes work once and expire within minutes. Run the script again and paste a fresh address.")
        print("Nothing was changed.")
        return 1
    if not gauth.credentials.refresh_token:
        print("\nGoogle signed you in but did not issue a refresh token, so the pipeline could not stay")
        print("signed in. Remove this app's access at https://myaccount.google.com/permissions and run")
        print("the script again. Nothing was changed.")
        return 1
    gauth.SaveCredentialsFile(str(target))
    print(f"\nSigned in. New credentials saved to {target.resolve()}")
    print("Testing them now ...\n")
    return 0


def verify(settings_file):
    check = GoogleAuth(settings_file=str(settings_file))
    try:
        check.LoadCredentials()
    except Exception as err:  # InvalidConfigError / InvalidCredentialsError
        print(f"Could not read the saved credentials: {err}")
        return 1
    creds = check.credentials
    if creds is None or not creds.refresh_token:
        print("No saved refresh token was found. Run:  python3 reauth_gdrive.py")
        return 1
    try:
        check.Refresh()  # the same refresh-token exchange that failed with invalid_grant
        check.Authorize()
        drive = GoogleDrive(check)
        about = drive.GetAbout()
        items = drive.ListFile({"q": "trashed=false", "maxResults": 3}).GetList()
    except RefreshError as err:
        print(f"Google refused the saved refresh token: {err}")
        print("Run (without --check):  python3 reauth_gdrive.py")
        print("If this recurs about a week after signing in, the OAuth consent screen is still in")
        print("'Testing' mode (7-day tokens): Google Cloud console > Google Auth platform > Audience > Publish app.")
        return 1
    except Exception as err:
        print(f"The token refreshed, but the Drive test call failed: {err!r}")
        return 1

    print("Google Drive access works.")
    email = (about.get("user") or {}).get("emailAddress")
    if email:
        print(f"  Signed in as : {email}")
    try:
        used, total = int(about["quotaBytesUsed"]), int(about["quotaBytesTotal"])
        print(f"  Drive storage: {_gb(used)} used" + (f" of {_gb(total)}" if total > 0 else ""))
    except (KeyError, ValueError, TypeError):
        pass
    print(f"  Listing test : OK ({len(items)} item{'s' if len(items) != 1 else ''} returned)")
    where = check.settings.get("save_credentials_file")
    if where:
        print(f"  Credentials  : {Path(where).resolve()}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="only test the saved credentials; do not sign in again",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run this on a computer WITH a browser (your laptop): it catches Google's redirect itself",
    )
    args = parser.parse_args(argv)

    root = find_repo_root()
    os.chdir(root)  # relative paths inside pydrive_settings.yaml are relative to the repo root
    secrets_dir = root / "secrets"
    settings_file = secrets_dir / "pydrive_settings.yaml"
    GoogleAuth.DEFAULT_SETTINGS["client_config_file"] = str(secrets_dir / "client_secrets.json")

    if not args.check:
        status = sign_in(settings_file, secrets_dir, local=args.local)
        if status:
            return status
    return verify(settings_file)


if __name__ == "__main__":
    sys.exit(main())