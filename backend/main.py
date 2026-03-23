from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import re
import json
import math
from dotenv import load_dotenv
from groq import Groq
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP")
SHOPIFY_TOKEN = os.getenv("SHOPIFY_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STORE_URL = os.getenv("STORE_URL", "https://ai-chatbot-lab.myshopify.com")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

DATA_DIR = "/data" if os.path.exists("/data") else "."
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge_base.json")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ─── Regex Patterns ───────────────────────────────────────────────────────────
GREET_RE      = re.compile(r'\b(hi|hello|hey|hii|helo|howdy|sup|yo)\b', re.I)
MORNING_RE    = re.compile(r'\b(good\s*(morning|evening|afternoon|night))\b', re.I)
HOW_ARE_YOU_RE= re.compile(r'\b(how are you|how r u|how are u|whats up|what\'s up)\b', re.I)
WHO_RE        = re.compile(r'\b(who are you|what are you|who r u)\b', re.I)
PRODUCT_RE    = re.compile(r'\b(products?|items?|collection|tshirts?|t-shirts?|shirts?|pants?|hoodies?|jackets?|bags?|caps?|hats?|shoes?|shorts|tanks?|compression|joggers?|sweatshirts?)\b', re.I)
PRICE_RE      = re.compile(r'\b(price|cost|rate|how much|charge|fee)\b', re.I)
ORDER_RE      = re.compile(r'\b(order|tracking|track|delivery|shipped|dispatch|status|where is my)\b', re.I)
COLOR_RE      = re.compile(r'\b(red|blue|black|white|green|grey|gray|pink|yellow|navy|brown|purple|maroon|orange|teal|olive|dusk|ash|camo|acid)\b', re.I)
SIZE_RE       = re.compile(r'\b(size|sizing|xs|small|medium|large|xl|xxl|2xl|3xl)\b', re.I)
DISCOUNT_RE   = re.compile(r'\b(discount|offer|sale|promo|coupon|deal|off)\b', re.I)
RETURN_RE     = re.compile(r'\b(return|refund|exchange|replace|cancel|policy)\b', re.I)
THANKS_RE     = re.compile(r'\b(thank|thanks|thankyou|thank you|thx|ty)\b', re.I)
HELP_RE       = re.compile(r'\b(help|support|assist|guide|faq)\b', re.I)
YES_RE        = re.compile(r'\b(yes+|yeah|yep|yup|sure|ok|okay|proceed|go ahead|show|display)\b', re.I)
NO_RE         = re.compile(r'\b(no|nope|nah|dont|don\'t|not)\b', re.I)
BYE_RE        = re.compile(r'\b(bye+|goodbye|see you|cya|tata|later|takcare|take care)\b', re.I)
LOVE_RE       = re.compile(r'\b(love|awesome|amazing|great|excellent|fantastic|wonderful|brilliant)\b', re.I)
NICE_RE       = re.compile(r'\b(nice|cool|good|superb|perfect|wow|impressive)\b', re.I)
BAD_RE        = re.compile(r'\b(bad|worst|terrible|horrible|poor|pathetic|useless|hate)\b', re.I)
OK_RE         = re.compile(r'\b(ok|okay|alright|got it|understood|i see|noted)\b', re.I)
YES_ONLY_RE   = re.compile(r'^(yes|yeah|yep|yup|sure|ok|okay)[\.\!]*$', re.I)
NO_ONLY_RE    = re.compile(r'^(no|nope|nah)[\.\!]*$', re.I)
CONTACT_RE    = re.compile(r'\b(contact|email|phone|call|reach|support|whatsapp)\b', re.I)
CART_RE       = re.compile(r'\b(cart|basket|bag|my cart|view cart|show cart)\b', re.I)

# ─── Session State ────────────────────────────────────────────────────────────
session_state = {}

# ─── Knowledge Base ───────────────────────────────────────────────────────────
def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as f:
            return json.load(f)
    return {"chunks": [], "initialized": False}

def save_knowledge(data):
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(data, f)

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

def compute_tfidf(query_tokens, doc_tokens, all_docs_tokens):
    scores = {}
    total_docs = len(all_docs_tokens)
    for token in query_tokens:
        tf = doc_tokens.count(token) / (len(doc_tokens) + 1)
        docs_with_token = sum(1 for d in all_docs_tokens if token in d)
        idf = math.log((total_docs + 1) / (docs_with_token + 1)) + 1
        scores[token] = tf * idf
    return sum(scores.values())

def search_knowledge(query, n_results=3):
    try:
        knowledge = load_knowledge()
        chunks = knowledge.get("chunks", [])
        if not chunks:
            return []
        texts = [c["text"] for c in chunks]
        query_tokens = tokenize(query)
        all_docs_tokens = [tokenize(t) for t in texts]
        scored = []
        for text, doc_tokens in zip(texts, all_docs_tokens):
            score = compute_tfidf(query_tokens, doc_tokens, all_docs_tokens)
            scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:n_results] if score > 0]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def scrape_store_website():
    """Scrape the store's own website for RAG"""
    try:
        knowledge = load_knowledge()
        if knowledge.get("initialized"):
            print("Knowledge base already initialized")
            return
        print(f"Initializing knowledge base from {STORE_URL}")
        visited = set()
        to_visit = [STORE_URL]
        all_chunks = []
        pages = 0
        while to_visit and pages < 10:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                res = requests.get(url, headers=HEADERS, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = re.sub(r'\s+', ' ', soup.get_text(separator=" ", strip=True))
                words = text.split()
                for i in range(0, len(words), 450):
                    chunk = " ".join(words[i:i+450])
                    if chunk:
                        all_chunks.append({"text": chunk, "url": url})
                pages += 1
                domain = f"{urlparse(STORE_URL).scheme}://{urlparse(STORE_URL).netloc}"
                for a in soup.find_all("a", href=True):
                    full = urljoin(url, a["href"])
                    if full.startswith(domain) and full not in visited:
                        to_visit.append(full)
            except Exception as e:
                print(f"Page error {url}: {e}")
        save_knowledge({"chunks": all_chunks, "initialized": True})
        print(f"Knowledge base ready: {len(all_chunks)} chunks from {pages} pages")
    except Exception as e:
        print(f"Scrape error: {e}")

# ─── Shopify API ──────────────────────────────────────────────────────────────
def get_shopify_products():
    try:
        url = f"https://{SHOPIFY_SHOP}/admin/api/2023-10/products.json?limit=50"
        res = requests.get(url, headers={"X-Shopify-Access-Token": SHOPIFY_TOKEN}, timeout=10)
        return res.json().get("products", [])
    except Exception as e:
        print(f"Shopify products error: {e}")
        return []

def get_shopify_orders_by_email(email):
    try:
        url = f"https://{SHOPIFY_SHOP}/admin/api/2023-10/orders.json?email={email}&status=any&limit=5"
        res = requests.get(url, headers={"X-Shopify-Access-Token": SHOPIFY_TOKEN}, timeout=10)
        return res.json().get("orders", [])
    except Exception as e:
        print(f"Shopify orders error: {e}")
        return []

def format_products_response(products):
    if not products:
        return {"type": "text", "text": "Sorry, I couldn't find any products matching that."}
    cards = []
    for p in products[:6]:
        image = None
        if p.get("images"):
            image = p["images"][0].get("src")
        variant = p.get("variants", [{}])[0]
        price = variant.get("price", "N/A")
        cards.append({
            "title": p["title"],
            "price": f"Rs. {price}",
            "image": image,
            "url": f"https://{SHOPIFY_SHOP}/products/{p['handle']}"
        })
    return {"type": "products", "products": cards}

def get_ai_answer(context, question):
    try:
        if groq_client:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"You are a helpful shopping assistant for this store. Answer based on this context:\n\n{context}\n\nBe concise and friendly. If you don't know, say so politely."},
                    {"role": "user", "content": question}
                ],
                max_tokens=200
            )
            return res.choices[0].message.content
    except Exception as e:
        print(f"AI error: {e}")
    return "I'm not sure about that. Please contact our support team for more details!"

# ─── Models ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    scrape_store_website()

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Shopify AI Chatbot running"}

@app.get("/debug-chat/{msg}")
def debug_chat(msg: str):
    return {
        "msg": msg,
        "PRODUCT_RE": bool(PRODUCT_RE.search(msg)),
        "GREET_RE": bool(GREET_RE.search(msg)),
        "PRICE_RE": bool(PRICE_RE.search(msg)),
    }
    
@app.get("/debug-shopify")
def debug_shopify():
    products = get_shopify_products()
    return {
        "total_products": len(products),
        "first_product": products[0]["title"] if products else "none"
    }

@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.strip()
    session_id = req.session_id
    state = session_state.get(session_id, {})

    # ── Waiting for email ──
    if state.get("waiting_email") and "@" in msg:
        orders = get_shopify_orders_by_email(msg)
        session_state[session_id] = {}
        if not orders:
            return {"type": "text", "text": f"No orders found for {msg}. Please check the email and try again."}
        lines = [f"📦 Here are your recent orders for {msg}:\n"]
        for o in orders:
            lines.append(f"Order #{o['order_number']} — {o['financial_status'].title()}")
            lines.append(f"Total: Rs. {o['total_price']}")
            lines.append(f"Status: {o.get('fulfillment_status') or 'Processing'}\n")
        return {"type": "text", "text": "\n".join(lines)}

    # ── Waiting for confirmation ──
    waiting = state.get("waiting_confirmation", "")

    if waiting == "show_products" and YES_RE.search(msg):
        session_state[session_id] = {}
        products = get_shopify_products()
        return format_products_response(products)

    if waiting == "show_products" and NO_RE.search(msg):
        session_state[session_id] = {}
        return {"type": "text", "text": "No problem! Let me know if you need anything else. 😊"}

    if waiting.startswith("show_single_product_"):
        title = waiting.replace("show_single_product_", "")
        if YES_RE.search(msg):
            session_state[session_id] = {}
            products = get_shopify_products()
            matched = [p for p in products if title.lower() in p["title"].lower()]
            return format_products_response(matched[:1])
        elif NO_RE.search(msg):
            session_state[session_id] = {}
            return {"type": "text", "text": "Alright! Let me know if you need anything else. 😊"}

    # ── Greetings ──
    if GREET_RE.search(msg):
        return {"type": "text", "text": "Hey there! 👋 Welcome! I can help you with products, prices, orders and more. What are you looking for?"}

    if MORNING_RE.search(msg):
        return {"type": "text", "text": "Good day! ☀️ How can I help you today?"}

    if HOW_ARE_YOU_RE.search(msg):
        return {"type": "text", "text": "I'm doing great, thanks for asking! 😊 How can I help you today?"}

    if WHO_RE.search(msg):
        return {"type": "text", "text": "I'm your store assistant! 🤖 I can help you find products, check prices, track orders and answer your questions."}

    # ── Positive/Negative reactions ──
    if LOVE_RE.search(msg) and not PRODUCT_RE.search(msg):
        return {"type": "text", "text": "That's so kind of you! 😊 Let me know if there's anything I can help you with!"}

    if NICE_RE.search(msg) and not PRODUCT_RE.search(msg):
        return {"type": "text", "text": "Thank you! 😊 Is there anything else I can help you with?"}

    if BAD_RE.search(msg):
        return {"type": "text", "text": "I'm sorry to hear that! 😔 Please contact our support team and we'll make it right."}

    if THANKS_RE.search(msg):
        return {"type": "text", "text": "You're welcome! 😊 Feel free to ask if you need anything else!"}

    if BYE_RE.search(msg):
        return {"type": "text", "text": "Goodbye! 👋 Have a great day! Come back anytime!"}

    # ── Cart ──
    if CART_RE.search(msg):
        return {"type": "cart"}
    
    # ── Price with product name ──
    if PRICE_RE.search(msg) and PRODUCT_RE.search(msg):
        products = get_shopify_products()
        stop = {"show","me","the","a","an","get","find","i","want","need","buy","some","any","all","products","items","price","cost","rate","much","how","is","of","what"}
        keywords = [k for k in msg.lower().split() if k not in stop and len(k) > 2]
        matched = []
        if keywords:
            matched = [p for p in products if any(k in p["title"].lower() for k in keywords)]
        # Also check color specifically
        if COLOR_RE.search(msg):
            color = COLOR_RE.search(msg).group(0).lower()
            color_matched = [p for p in products if color in p["title"].lower()]
            if color_matched:
                matched = color_matched
        if matched:
            p = matched[0]
            price = p.get("variants", [{}])[0].get("price", "N/A")
            session_state[session_id] = {"waiting_confirmation": f"show_single_product_{p['title']}"}
            return {"type": "text", "text": f"The {p['title']} is priced at Rs. {price}. Would you like to see the product? 😊"}

    # ── Price only ──
    if PRICE_RE.search(msg) and not PRODUCT_RE.search(msg):
        if COLOR_RE.search(msg):
            color = COLOR_RE.search(msg).group(0)
            products = get_shopify_products()
            matched = [p for p in products if color.lower() in p["title"].lower()]
            if matched:
                p = matched[0]
                price = p.get("variants", [{}])[0].get("price", "N/A")
                session_state[session_id] = {"waiting_confirmation": f"show_single_product_{p['title']}"}
                return {"type": "text", "text": f"The {p['title']} is priced at Rs. {price}. Would you like to see the product? 😊"}
        session_state[session_id] = {"waiting_confirmation": "show_products"}
        return {"type": "text", "text": "Our products are competitively priced! Would you like me to show you all products with prices? 😊"}
        
    # ── Products with color ──
    if COLOR_RE.search(msg) and PRODUCT_RE.search(msg):
        color = COLOR_RE.search(msg).group(0)
        products = get_shopify_products()
        matched = [p for p in products if color.lower() in p["title"].lower() or
                   any(color.lower() in str(v.get("title","")).lower() for v in p.get("variants",[]))]
        if matched:
            return format_products_response(matched)
        return {"type": "text", "text": f"Sorry, I couldn't find any {color} products. Want to see all products instead?"}

    # ── All products ──
    print(f"PRODUCT_RE check: {bool(PRODUCT_RE.search(msg))} for msg: {msg}")
    if PRODUCT_RE.search(msg):
        products = get_shopify_products()
        keyword = msg.lower()
        stop = {"show","me","the","a","an","get","find","i","want","need","buy","some","any","all","products","items"}
        keywords = [k for k in keyword.split() if k not in stop and len(k) > 2]
        if keywords:
            matched = [p for p in products if any(k in p["title"].lower() for k in keywords)]
            if matched:
                return format_products_response(matched)
        return format_products_response(products)


    # ── Order tracking ──
    if ORDER_RE.search(msg):
        session_state[session_id] = {"waiting_email": True}
        return {"type": "text", "text": "I'd be happy to help track your order! 📦 Please share your email address used during purchase."}

    # ── Discount ──
    if DISCOUNT_RE.search(msg):
        return {"type": "text", "text": "🎉 We regularly offer discounts and seasonal sales! Follow us on social media or subscribe to our newsletter to stay updated on the latest deals!"}

    # ── Return policy ──
    if RETURN_RE.search(msg):
        return {"type": "text", "text": "↩️ We have a hassle-free return policy! You can return products within 7 days of delivery. Please contact our support team to initiate a return."}

    # ── Size ──
    if SIZE_RE.search(msg):
        return {"type": "text", "text": "📏 We offer sizes XS, S, M, L, XL, XXL and 3XL. Each product page has a detailed size chart. Need help with a specific product?"}

    # ── Contact ──
    if CONTACT_RE.search(msg):
        return {"type": "text", "text": "📞 You can reach us at:\n📧 Email: support@ai-chatbot-lab.com\n💬 WhatsApp: +91-XXXXXXXXXX\n⏰ Available: Mon-Sat, 10AM-6PM"}

    # ── Help ──
    if HELP_RE.search(msg):
        return {"type": "text", "text": "I can help you with:\n🛍️ Browse products\n💰 Check prices\n📦 Track orders\n↩️ Returns & exchanges\n📏 Size guide\n📞 Contact support\n\nWhat do you need?"}

    if OK_RE.search(msg) and len(msg.split()) <= 3:
        return {"type": "text", "text": "Great! Let me know if you need anything else. 😊"}

    if YES_ONLY_RE.search(msg):
        return {"type": "text", "text": "Sure! What would you like to know? 😊"}

    if NO_ONLY_RE.search(msg):
        return {"type": "text", "text": "No problem! Feel free to ask anything else. 😊"}

    # ── RAG Fallback ──
    print(f"Reached RAG fallback for msg: {msg}")
    chunks = search_knowledge(msg)
    if chunks:
        context = "\n\n".join(chunks)
        answer = get_ai_answer(context, msg)
        return {"type": "text", "text": answer}

    return {"type": "text", "text": "I'm not sure about that. Can you rephrase or ask something else? I can help with products, orders, prices and more! 😊"}