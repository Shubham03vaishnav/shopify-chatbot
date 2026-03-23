# 🤖 Shopify AI Chatbot

An intelligent AI-powered chatbot for Shopify stores that helps customers find products, track orders, check prices and get instant support.

## 🌟 Live Demo
**Store:** [ai-chatbot-lab.myshopify.com](https://ai-chatbot-lab.myshopify.com)  
**Backend API:** [shopify-chatbot-production-ce75.up.railway.app](https://shopify-chatbot-production-ce75.up.railway.app)

## ✨ Features

- 🛍️ **Product Search** — Find products by name, type or color
- 💰 **Price Queries** — Check prices with confirmation flow
- 📦 **Order Tracking** — Track orders by email
- 🛒 **Cart Management** — View and remove cart items
- ↩️ **Return Policy** — Instant policy information
- 📏 **Size Guide** — Size recommendations
- 📞 **Contact Support** — Store contact information
- 🧠 **AI Powered** — Groq LLM for intelligent responses
- 💬 **Natural Language** — Understands casual conversation

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| AI Model | Groq — LLaMA 3.3 70B |
| NLP | Regex + RAG System |
| Database | Railway Volume (JSON) |
| Deployment | Railway |
| Frontend | Vanilla JS embedded in Shopify |
| Store Platform | Shopify |

## 🏗️ Architecture
```
Customer Message
      ↓
Regex Pattern Matching (fast, free)
      ↓
Shopify API (products, orders)
      ↓
RAG System (store knowledge)
      ↓
Groq AI (intelligent fallback)
      ↓
Response to Customer
```

## 🚀 How It Works

1. Customer opens chatbot on Shopify store
2. Message is sent to FastAPI backend
3. Regex patterns handle common queries instantly
4. Shopify API fetches real product and order data
5. RAG system searches store knowledge for complex questions
6. Groq AI generates intelligent responses as fallback

## 📦 Installation

### Backend Setup

1. Clone the repo
```bash
git clone https://github.com/Shubham03vaishnav/shopify-chatbot.git
cd shopify-chatbot/backend
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create `.env` file
```env
SHOPIFY_SHOP=your-store.myshopify.com
SHOPIFY_TOKEN=your_shopify_token
GROQ_API_KEY=your_groq_key
STORE_URL=https://your-store.myshopify.com
```

4. Run the server
```bash
uvicorn main:app --reload
```

### Shopify Integration

1. Go to your Shopify admin
2. Online Store → Themes → Edit Code
3. Open `layout/theme.liquid`
4. Add the chatbot script before `</body>`
5. Update `API_URL` with your backend URL

## 🔧 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/chat` | POST | Main chat endpoint |
| `/debug-shopify` | GET | Test Shopify connection |

## 📱 Chat Examples
```
User: show me products
Bot: Here are some products for you! 🛍️ [product cards]

User: price of black tshirt
Bot: The Black Loose Fit Tshirt is priced at Rs. 599.00. Would you like to see the product?

User: track my order
Bot: Please share your email address used during purchase.

User: return policy
Bot: We have a hassle-free return policy! You can return products within 7 days.
```

## 🙋 Author

**Shubham Vaishnav**  
[GitHub](https://github.com/Shubham03vaishnav) · [LinkedIn](https://www.linkedin.com/in/shubham-vaishnav-b79133271)

## 📄 License

MIT License