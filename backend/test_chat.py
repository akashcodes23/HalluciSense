import requests

# 1. Login with the user's email if possible, or create a new user to get token
# Wait, let's just create a new user to get the token
res = requests.post("http://127.0.0.1:8000/api/v1/auth/register", json={
    "full_name": "Test User 2",
    "email": "test2@example.com",
    "password": "StrongPassword1!"
})
if res.status_code == 409:
    res = requests.post("http://127.0.0.1:8000/api/v1/auth/login", json={
        "email": "test2@example.com",
        "password": "StrongPassword1!"
    })
print("Login status:", res.status_code)
token = res.json().get("tokens", {}).get("access_token")

headers = {"Authorization": f"Bearer {token}"}
# 2. Create a chat
res = requests.post("http://127.0.0.1:8000/api/v1/chats/", headers=headers, json={
    "title": "Test Chat"
})
print("Create chat status:", res.status_code)
if res.status_code != 200:
    print(res.text)

chat_id = res.json().get("id")

# 3. Send message
res = requests.post(f"http://127.0.0.1:8000/api/v1/chats/{chat_id}/messages", headers=headers, json={
    "content": "who is the pm of india",
    "mode": "VERIFY"
})
print("Send message status:", res.status_code)
print(res.text)

