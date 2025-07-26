# setup_env.py
import os
import getpass

def create_env_file():
    """
    Interactively prompts the user for all API credentials and generates a .env file.
    """
    print("--- Interactive API Credential Setup ---")
    print("This script will create a .env file to securely store your API keys.")
    print("-" * 40)

    if os.path.exists('.env'):
        overwrite = input("A .env file already exists. Do you want to overwrite it? (y/n): ").lower()
        if overwrite != 'y':
            print("Setup cancelled. Your existing .env file has not been changed.")
            return

    try:
        # --- Collect Credentials ---
        print("\n--- Reddit Credentials ---")
        reddit_client_id = input("Enter your Reddit Client ID: ")
        reddit_client_secret = getpass.getpass("Enter your Reddit Client Secret: ")
        reddit_user_agent = input("Enter your Reddit User Agent (e.g., MyBot/1.0 by u/YourUsername): ")
        
        print("\n--- Google Gemini Credentials ---")
        gemini_api_key = getpass.getpass("Enter your Google Gemini API Key: ")

        print("\n--- Bluesky Credentials ---")
        bluesky_handle = input("Enter your Bluesky Handle (e.g., your-name.bsky.social): ")
        bluesky_app_password = getpass.getpass("Enter your Bluesky App Password: ")
        
        # --- Write to .env file ---
        with open('.env', 'w') as f:
            f.write("# Reddit API Credentials\n")
            f.write(f'REDDIT_CLIENT_ID="{reddit_client_id}"\n')
            f.write(f'REDDIT_CLIENT_SECRET="{reddit_client_secret}"\n')
            f.write(f'REDDIT_USER_AGENT="{reddit_user_agent}"\n\n')
            
            f.write("# Google Gemini API Key\n")
            f.write(f'GEMINI_API_KEY="{gemini_api_key}"\n\n')

            f.write("# Bluesky Credentials\n")
            f.write(f'BLUESKY_HANDLE="{bluesky_handle}"\n')
            f.write(f'BLUESKY_APP_PASSWORD="{bluesky_app_password}"\n')

        print("\n" + "-" * 40)
        print(".env file has been created with all credentials.")
        print("-" * 40)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    create_env_file()
