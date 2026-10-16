
import os
import re
import json
import sqlite3
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify, session, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
DB_PATH = os.path.join(BASE_DIR, "find.db")

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__, static_folder=PUBLIC_DIR)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-local-secret")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)


def save_history(query):
    user_id = session.get("user_id")
    query = (query or "").strip()

    if not user_id or not query:
        return

    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO history (user_id, query, created_at) VALUES (?, ?, ?)",
                (user_id, query[:1000], datetime.now().isoformat(timespec="seconds"))
            )
    except sqlite3.Error as error:
        print("History database error:", error)


init_db()

# --------------------------------------------------
# WEB SEARCH
# --------------------------------------------------

def web_search(query, max_results=10):
    """Search web pages through DuckDuckGo HTML results."""
    query = (query or "").strip()
    if not query:
        return []

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=15
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        seen = set()

        for item in soup.select(".result"):
            link = item.select_one(".result__title a")

            if not link:
                link = item.select_one("a.result__a")

            if not link:
                continue

            href = link.get("href", "").strip()
            title = link.get_text(" ", strip=True)

            if "duckduckgo.com/l/" in href:
                target = parse_qs(urlparse(href).query).get("uddg", [""])[0]
                if target:
                    href = unquote(target)

            if not href.startswith(("http://", "https://")):
                continue

            if href in seen:
                continue

            seen.add(href)

            snippet_node = item.select_one(
                ".result__snippet, .result-snippet"
            )

            results.append({
                "title": title or href,
                "url": href,
                "link": href,
                "snippet": (
                    snippet_node.get_text(" ", strip=True)
                    if snippet_node else ""
                )
            })

            if len(results) >= max_results:
                break

        print(f"Web search: {len(results)} results for {query!r}")
        return results

    except requests.RequestException as error:
        print("Web search request failed:", repr(error))
        return []


# --------------------------------------------------
# IMAGE SEARCH
# --------------------------------------------------

def image_search(query, max_results=30):
    """
    Best-effort DuckDuckGo image search.
    Returns direct image URLs and source-page URLs when available.
    This is an unofficial provider interface and may change.
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        # First obtain the image-search token.
        response = requests.get(
            "https://duckduckgo.com/",
            params={"q": query, "iax": "images", "ia": "images"},
            headers=HEADERS,
            timeout=15
        )
        response.raise_for_status()

        match = re.search(r'vqd=["\']([^"\']+)["\']', response.text)

        if not match:
            match = re.search(r"vqd=([\d-]+)", response.text)

        if not match:
            print("Image search: provider did not return a search token.")
            return []

        vqd = match.group(1)

        # Then request image results.
        image_response = requests.get(
            "https://duckduckgo.com/i.js",
            params={
                "l": "us-en",
                "o": "json",
                "q": query,
                "vqd": vqd,
                "f": ",,,",
                "p": "1"
            },
            headers={
                **HEADERS,
                "Referer": "https://duckduckgo.com/"
            },
            timeout=20
        )
        image_response.raise_for_status()

        payload = image_response.json()
        items = payload.get("results", [])
        results = []
        seen = set()

        for item in items:
            image_url = item.get("image") or item.get("thumbnail")
            thumbnail = item.get("thumbnail") or image_url
            source_url = item.get("url") or item.get("source")

            if not image_url:
                continue

            if not image_url.startswith(("https://", "http://")):
                continue

            if image_url in seen:
                continue

            seen.add(image_url)

            results.append({
                "title": item.get("title") or query,
                "image": image_url,
                "image_url": image_url,
                "thumbnail": thumbnail,
                "url": source_url or image_url,
                "source": item.get("source") or "",
                "width": item.get("width"),
                "height": item.get("height")
            })

            if len(results) >= max_results:
                break

        print(f"Image search: {len(results)} results for {query!r}")
        return results

    except (requests.RequestException, ValueError, TypeError) as error:
        print("Image search failed:", repr(error))
        return []


# --------------------------------------------------
# GROQ AI
# --------------------------------------------------

def ask_groq(messages, temperature=0.5):
    if not GROQ_API_KEY:
        return {
            "ok": False,
            "error": "GROQ_API_KEY is missing from your .env file."
        }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": temperature
            },
            timeout=60
        )

        if response.status_code >= 400:
            print("Groq API error:", response.status_code, response.text[:500])
            return {
                "ok": False,
                "error": f"AI provider returned HTTP {response.status_code}."
            }

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return {"ok": True, "answer": answer}

    except (requests.RequestException, ValueError, KeyError, IndexError) as error:
        print("AI request failed:", repr(error))
        return {
            "ok": False,
            "error": "Could not get an AI response. Please try again."
        }


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(PUBLIC_DIR, "Find.html")


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "app": "Find",
        "ai_configured": bool(GROQ_API_KEY)
    })


# --------------------------------------------------
# SIGNUP / LOGIN
# --------------------------------------------------

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not name or not email or not password:
        return jsonify({
            "ok": False,
            "error": "Name, email and password are required."
        }), 400

    if len(password) < 8:
        return jsonify({
            "ok": False,
            "error": "Password must contain at least 8 characters."
        }), 400

    try:
        with get_db() as db:
            cursor = db.execute(
                """INSERT INTO users (name, email, password, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    name,
                    email,
                    generate_password_hash(password),
                    datetime.now().isoformat(timespec="seconds")
                )
            )
            user_id = cursor.lastrowid

        session["user_id"] = user_id
        session["user_name"] = name
        session["user_email"] = email

        return jsonify({
            "ok": True,
            "user": {"id": user_id, "name": name, "email": email}
        })

    except sqlite3.IntegrityError:
        return jsonify({
            "ok": False,
            "error": "An account with this email already exists."
        }), 409


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    with get_db() as db:
        user = db.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

    if not user or not check_password_hash(user["password"], password):
        return jsonify({
            "ok": False,
            "error": "Incorrect email or password."
        }), 401

    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    session["user_email"] = user["email"]

    return jsonify({
        "ok": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }
    })


@app.route("/api/me")
def me():
    if not session.get("user_id"):
        return jsonify({"logged_in": False, "user": None})

    return jsonify({
        "logged_in": True,
        "user": {
            "id": session.get("user_id"),
            "name": session.get("user_name"),
            "email": session.get("user_email")
        }
    })


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


# --------------------------------------------------
# SEARCH ROUTES
# --------------------------------------------------

@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"results": [], "query": ""})

    results = web_search(query, 10)
    save_history(query)

    return jsonify({
        "query": query,
        "results": results,
        "count": len(results),
        "message": "" if results else (
            "No web results were returned. The search provider may be "
            "temporarily unavailable."
        )
    })



@app.route("/api/images")
def images():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "query": "",
            "results": [],
            "images": [],
            "error": "Enter a search query."
        }), 400

    import json
    import urllib.parse
    import urllib.request

    try:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 30,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 500,
            "format": "json"
        }

        api_url = (
            "https://commons.wikimedia.org/w/api.php?"
            + urllib.parse.urlencode(params)
        )

        req = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": "FindSearchPrototype/1.0"
            }
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})
        results = []

        for page in pages.values():
            image_info = page.get("imageinfo", [])

            if not image_info:
                continue

            info = image_info[0]
            image_url = info.get("thumburl") or info.get("url")

            if not image_url:
                continue

            title = page.get("title", "Image").replace("File:", "")
            page_url = (
                "https://commons.wikimedia.org/?curid="
                + str(page.get("pageid", ""))
            )

            results.append({
                "title": title,
                "image": image_url,
                "image_url": image_url,
                "thumbnail": image_url,
                "thumbnail_url": image_url,
                "src": image_url,
                "url": page_url,
                "page_url": page_url,
                "description": title,
                "snippet": title
            })

        save_history("Images: " + query)

        return jsonify({
            "query": query,
            "results": results,
            "images": results,
            "count": len(results),
            "message": "" if results else (
                "No images found. Try a different search."
            )
        })

    except Exception as e:
        return jsonify({
            "query": query,
            "results": [],
            "images": [],
            "count": 0,
            "message": "Image search failed. Check your internet connection.",
            "error": str(e)
        }), 502


@app.route("/api/news")
def news():
    query = request.args.get("q", "").strip() or "latest news"
    results = web_search(query + " news", 10)
    save_history("News: " + query)

    return jsonify({
        "query": query,
        "results": results,
        "news": results,
        "count": len(results)
    })



@app.route("/api/videos")
def videos():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "query": "",
            "results": [],
            "videos": [],
            "count": 0
        })

    import urllib.parse
    import urllib.request
    import re
    import html

    try:
        search_url = (
            "https://www.youtube.com/results?search_query="
            + urllib.parse.quote_plus(query)
        )

        req = urllib.request.Request(
            search_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            page = response.read().decode("utf-8", errors="ignore")

        # Extract video IDs and titles from YouTube's search page.
        pattern = re.compile(
            r'"videoRenderer":\{"videoId":"([^"]+)".*?'
            r'"title":\{"runs":\[\{"text":"([^"]+)"',
            re.DOTALL
        )

        results = []
        seen = set()

        for match in pattern.finditer(page):
            video_id = match.group(1)

            if video_id in seen:
                continue

            seen.add(video_id)
            title = html.unescape(match.group(2))
            title = title.replace("\\u0026", "&")
            title = title.replace("\\\"", '"')

            results.append({
                "title": title,
                "url": "https://www.youtube.com/watch?v=" + video_id,
                "video_url": "https://www.youtube.com/watch?v=" + video_id,
                "thumbnail": (
                    "https://i.ytimg.com/vi/"
                    + video_id
                    + "/hqdefault.jpg"
                ),
                "thumbnail_url": (
                    "https://i.ytimg.com/vi/"
                    + video_id
                    + "/hqdefault.jpg"
                ),
                "source": "YouTube"
            })

            if len(results) >= 20:
                break

        save_history("Videos: " + query)

        return jsonify({
            "query": query,
            "results": results,
            "videos": results,
            "count": len(results),
            "message": "" if results else (
                "No videos returned. YouTube may have changed its page "
                "or blocked the request."
            )
        })

    except Exception as e:
        return jsonify({
            "query": query,
            "results": [],
            "videos": [],
            "count": 0,
            "message": "Video search failed. Check your internet connection.",
            "error": str(e)
        }), 502

# --------------------------------------------------
# AI MODE
# --------------------------------------------------

@app.route("/api/ask", methods=["GET", "POST"])
def ask():
    data = request.get_json(silent=True) or {}

    query = (
        data.get("message")
        or data.get("query")
        or request.args.get("q", "")
    )
    query = str(query).strip()

    if not query:
        return jsonify({
            "ok": False,
            "error": "Enter a question."
        }), 400

    
    messages = [{
        "role": "system",
        "content": (
            "You are Find, a helpful AI search assistant. "
            "Answer clearly and accurately. If uncertain, say so. "
            "Use headings and lists when useful. Do not invent sources. "

            "Find was created by and is owned by Samyak Jain. "
            "He is a Class 7 student from Indore, Madhya Pradesh, India, "
            "and studies at Choithram School, Manik Bagh. "
            "He is also the owner of the Secret Developer website. "

            "When users ask 'Who made you?', 'Who made Find?', "
            "'Who created Find?', 'Who owns Find?', 'Search Samyak Jain', "
            "'Who is Samyak Jain?', or ask about Secret Developer, "
            "share this creator information clearly. "
            "You are Find, not ChatGPT."
        )
    }]

    # Accept recent conversation history from the frontend.
    history = data.get("history", [])

    if isinstance(history, list):
        for item in history[-10:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content") or item.get("message")

            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content[:6000]
                })

    # Add the current question unless already included.
    if not messages or messages[-1].get("content") != query:
        messages.append({"role": "user", "content": query})

    result = ask_groq(messages)
    save_history("AI: " + query)

    if not result["ok"]:
        return jsonify({
            "ok": False,
            "answer": "",
            "error": result["error"]
        }), 502

    return jsonify({
        "ok": True,
        "answer": result["answer"],
        "response": result["answer"],
        "message": result["answer"],
        "query": query
    })


@app.route("/api/research", methods=["GET", "POST"])
def research():
    data = request.get_json(silent=True) or {}
    query = str(
        data.get("query")
        or data.get("message")
        or request.args.get("q", "")
    ).strip()

    if not query:
        return jsonify({
            "ok": False,
            "error": "Enter a research topic."
        }), 400

    results = web_search(query, 8)

    source_text = "\n\n".join(
        f"{i + 1}. {item['title']}\n"
        f"URL: {item['url']}\n"
        f"Details: {item['snippet']}"
        for i, item in enumerate(results)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Find Research. Write a balanced, useful research "
                "summary. Use only the supplied search results for sourced "
                "claims. Cite sources by their numbered result and URL. "
                "Clearly mention when the sources are insufficient."
            )
        },
        {
            "role": "user",
            "content": (
                f"Research topic: {query}\n\n"
                f"Search results:\n{source_text or 'No web results returned.'}"
            )
        }
    ]

    ai = ask_groq(messages)
    save_history("Research: " + query)

    if not ai["ok"]:
        return jsonify({
            "ok": False,
            "error": ai["error"],
            "results": results
        }), 502

    return jsonify({
        "ok": True,
        "query": query,
        "answer": ai["answer"],
        "response": ai["answer"],
        "results": results,
        "sources": results
    })


# --------------------------------------------------
# SEARCH HISTORY
# --------------------------------------------------

@app.route("/api/history", methods=["GET", "DELETE"])
def history():
    user_id = session.get("user_id")

    if request.method == "DELETE":
        if not user_id:
            return jsonify({
                "ok": False,
                "error": "Log in to clear saved history."
            }), 401

        with get_db() as db:
            db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))

        return jsonify({"ok": True})

    if not user_id:
        return jsonify({"history": [], "results": []})

    with get_db() as db:
        rows = db.execute(
            """SELECT id, query, created_at
               FROM history WHERE user_id = ?
               ORDER BY id DESC LIMIT 100""",
            (user_id,)
        ).fetchall()

    items = [
        {
            "id": row["id"],
            "query": row["query"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]

    return jsonify({"history": items, "results": items})


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":
    print("Find server is starting...")
    print("Open http://localhost:3000 in your browser.")
    print("AI configured:", bool(GROQ_API_KEY))

    app.run(host="127.0.0.1", port=3000, debug=True)
