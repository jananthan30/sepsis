"""
Authenticate with Google Cloud for BigQuery access.
This will open a browser window for you to log in.
"""
import pydata_google_auth
import os

print("=" * 60)
print("Google Cloud Authentication for BigQuery")
print("=" * 60)
print()
print("A browser window will open for you to log in with your")
print("Google account that has access to MIMIC-IV BigQuery data.")
print()

# Get credentials with BigQuery scope
credentials = pydata_google_auth.get_user_credentials(
    scopes=['https://www.googleapis.com/auth/bigquery'],
    auth_local_webserver=True
)

# Save credentials to the default location
credentials_path = os.path.join(
    os.environ.get('APPDATA', os.path.expanduser('~')),
    'gcloud',
    'application_default_credentials.json'
)

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(credentials_path), exist_ok=True)

# pydata_google_auth caches the credentials, but we need to save them in the right format
# for google-cloud-bigquery to find them
import json

# Create the credentials file
creds_data = {
    "client_id": credentials.client_id,
    "client_secret": credentials.client_secret,
    "refresh_token": credentials.refresh_token,
    "type": "authorized_user"
}

with open(credentials_path, 'w') as f:
    json.dump(creds_data, f)

print()
print("=" * 60)
print("SUCCESS! Credentials saved to:")
print(f"  {credentials_path}")
print()
print("You can now run the training script:")
print("  python train_model_v3.py --extract --project sepsis-prediction-2025")
print("=" * 60)
