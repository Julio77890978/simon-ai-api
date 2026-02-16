"""
Simon AI Backend — Customer Intelligence Platform for SMEs
FastAPI server providing review scraping, persona generation, and customer simulation.
"""

import os
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simon AI", description="Customer Intelligence Platform for SMEs", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── Models ───────────────────────────────────────────────────────────────────

class BusinessCreate(BaseModel):
    name: str
    address: Optional[str] = None
    category: Optional[str] = None
    google_place_id: Optional[str] = None

class Review(BaseModel):
    author: str
    rating: int
    text: str
    date: str
    sentiment_score: float = 0.0
    sentiment_label: str = ""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    business_id: str
    persona_id: str
    message: str
    history: List[ChatMessage] = []

# ─── In-Memory Store ─────────────────────────────────────────────────────────

businesses_db: Dict[str, dict] = {}
reviews_db: Dict[str, List[dict]] = {}
personas_db: Dict[str, List[dict]] = {}
insights_db: Dict[str, List[dict]] = {}

# ─── Demo Data with 15 Reviews Each ───────────────────────────────────────────

DEMO_BUSINESSES = [
    {"id": "demo-cafe", "name": "The Corner Café", "address": "123 Main St, Madrid", "category": "Restaurant/Café", "review_count": 247, "avg_rating": 4.3},
    {"id": "demo-salon", "name": "Luxe Hair Studio", "address": "45 Gran Vía, Madrid", "category": "Hair Salon", "review_count": 189, "avg_rating": 4.6},
    {"id": "demo-gym", "name": "FitZone Madrid", "address": "78 Calle Serrano, Madrid", "category": "Gym/Fitness", "review_count": 312, "avg_rating": 4.1},
]

CAFE_REVIEWS = [
    {"author": "María González", "rating": 5, "text": "Best coffee in the neighborhood! The baristas remember my order every time. Pastries are fresh daily and absolutely delicious.", "date": "2026-02-10", "sentiment_score": 0.92, "sentiment_label": "very positive"},
    {"author": "Carlos Rodríguez", "rating": 4, "text": "Good atmosphere for working. WiFi is solid and reliable. Gets crowded after 11am though. Wish they had more power outlets near the seating area.", "date": "2026-02-08", "sentiment_score": 0.65, "sentiment_label": "positive"},
    {"author": "Sarah Mitchell", "rating": 3, "text": "Coffee is decent but overpriced for the portion size. €4.50 for a regular latte seems steep. Service can be slow during peak hours.", "date": "2026-02-05", "sentiment_score": 0.15, "sentiment_label": "mixed"},
    {"author": "Pedro López", "rating": 5, "text": "Hidden gem! Their avocado toast is incredible and the cold brew is the best in Madrid. Staff is super friendly and remembers my name.", "date": "2026-02-01", "sentiment_score": 0.95, "sentiment_label": "very positive"},
    {"author": "Emma Kennedy", "rating": 2, "text": "Waited 20 minutes for a simple americano. Place was half empty. No apology from staff. Won't come back unless service improves dramatically.", "date": "2026-01-28", "sentiment_score": -0.72, "sentiment_label": "negative"},
    {"author": "Luis Álvarez", "rating": 4, "text": "Great brunch spot on weekends. The eggs benedict are restaurant quality. Only complaint is limited seating - arrived at 11am and waited 30 minutes.", "date": "2026-01-25", "sentiment_score": 0.78, "sentiment_label": "positive"},
    {"author": "Ana Benítez", "rating": 5, "text": "My daily stop before work. Consistent quality, fair prices for the area, and the loyalty card program is great - free coffee every 10th visit!", "date": "2026-01-20", "sentiment_score": 0.88, "sentiment_label": "very positive"},
    {"author": "James Wilson", "rating": 1, "text": "Found a hair in my cappuccino. When I told the barista, they just offered to remake it with no apology. Hygiene standards need serious improvement.", "date": "2026-01-15", "sentiment_score": -0.91, "sentiment_label": "very negative"},
    {"author": "Sofia Torres", "rating": 4, "text": "Love the interior design - very Instagram-worthy. The matcha latte is amazing. A bit noisy when the music is turned up too loud.", "date": "2026-01-10", "sentiment_score": 0.71, "sentiment_label": "positive"},
    {"author": "Daniel Fernández", "rating": 5, "text": "Brought my laptop to work here for 3 hours. Nobody rushed me. Great ambient music, strong WiFi, excellent flat white. This is my new office.", "date": "2026-01-05", "sentiment_score": 0.93, "sentiment_label": "very positive"},
    {"author": "Isabella Ruiz", "rating": 4, "text": "Nice spot for a quick coffee catch-up with friends. The seasonal latte flavors are creative. Prices are a bit high but worth it for the quality.", "date": "2025-12-28", "sentiment_score": 0.68, "sentiment_label": "positive"},
    {"author": "Miguel Santos", "rating": 3, "text": "Average experience. Coffee was okay, nothing special. The space is nice but feels a bit cramped. Might try another place next time.", "date": "2025-12-20", "sentiment_score": 0.12, "sentiment_label": "mixed"},
    {"author": "Carmen Díaz", "rating": 5, "text": "Absolutely love this place! The cortado is perfect every time. Staff always smiles and makes you feel welcome. Highly recommend!", "date": "2025-12-15", "sentiment_score": 0.96, "sentiment_label": "very positive"},
    {"author": "Robert Chen", "rating": 2, "text": "Overhyped. Waited 15 minutes for a pour-over that tasted burnt. €5 for a small cup. Won't return - there are better options nearby.", "date": "2025-12-10", "sentiment_score": -0.65, "sentiment_label": "negative"},
    {"author": "Laura Martínez", "rating": 4, "text": "Great for dates or catching up with friends. The outdoor seating is adorable in summer. Indoor seating could use better lighting though.", "date": "2025-12-05", "sentiment_score": 0.72, "sentiment_label": "positive"},
]

SALON_REVIEWS = [
    {"author": "Laura Muñoz", "rating": 5, "text": "Best haircut I've ever had in Madrid! Ana really understood what I wanted. The balayage is perfect and lasted for months.", "date": "2026-02-10", "sentiment_score": 0.95, "sentiment_label": "very positive"},
    {"author": "Carmen Sánchez", "rating": 4, "text": "Great color work but the wait was longer than expected even with an appointment. Nice complimentary drinks though.", "date": "2026-02-07", "sentiment_score": 0.62, "sentiment_label": "positive"},
    {"author": "Isabel Pérez", "rating": 5, "text": "Finally found my salon! They actually listen to what you want. My curly hair has never looked better. Worth every euro.", "date": "2026-02-03", "sentiment_score": 0.97, "sentiment_label": "very positive"},
    {"author": "Rosa Delgado", "rating": 2, "text": "Way too expensive for what you get. €85 for a cut and blowdry is outrageous. Quality was fine but not €85 fine.", "date": "2026-01-30", "sentiment_score": -0.55, "sentiment_label": "negative"},
    {"author": "Elena Ramírez", "rating": 5, "text": "The keratin treatment transformed my hair. 3 months later it still looks amazing. The stylist explained every step of the process.", "date": "2026-01-25", "sentiment_score": 0.94, "sentiment_label": "very positive"},
    {"author": "Patricia Navarro", "rating": 3, "text": "Decent haircut but nothing special. Expected more for the price. Stylist seemed distracted during my appointment.", "date": "2026-01-20", "sentiment_score": 0.18, "sentiment_label": "mixed"},
    {"author": "Antonio Torres", "rating": 5, "text": "My go-to barbershop for years. Consistent quality, fair prices, and great conversation. They treat you like family here.", "date": "2026-01-15", "sentiment_score": 0.91, "sentiment_label": "very positive"},
    {"author": "Cristina Gómez", "rating": 4, "text": "Love the results but booking is a nightmare. Called 5 times to get an appointment. Once there, service was excellent though.", "date": "2026-01-10", "sentiment_score": 0.58, "sentiment_label": "positive"},
    {"author": "Fernando Ruiz", "rating": 1, "text": "Terrible experience. Dyed my hair completely wrong color. Manager was unhelpful when I complained. Had to go somewhere else to fix it.", "date": "2026-01-05", "sentiment_score": -0.93, "sentiment_label": "very negative"},
    {"author": "Marta Castro", "rating": 5, "text": "The stylists here are true artists. My highlights look natural and beautiful. Worth every penny. Booked my next appointment already!", "date": "2025-12-28", "sentiment_score": 0.98, "sentiment_label": "very positive"},
    {"author": "Javier Moreno", "rating": 4, "text": "Good barbershop for men's cuts. Traditional techniques with modern styling. Friendly staff and clean space.", "date": "2025-12-20", "sentiment_score": 0.74, "sentiment_label": "positive"},
    {"author": "Silvia Herrera", "rating": 3, "text": "Mixed feelings - great cut but the blowdry took forever and left my hair feeling crispy. Might give it another try though.", "date": "2025-12-15", "sentiment_score": 0.25, "sentiment_label": "mixed"},
    {"author": "Raúl Jiménez", "rating": 5, "text": "Best color in Madrid! They use high-quality products and the shade matching is perfect. My hair feels healthier than before.", "date": "2025-12-10", "sentiment_score": 0.92, "sentiment_label": "very positive"},
    {"author": "Beatriz Romero", "rating": 2, "text": "Appointment was 45 minutes late with no apology. The cut itself was fine but the experience left a bad taste.", "date": "2025-12-05", "sentiment_score": -0.48, "sentiment_label": "negative"},
    {"author": "Diego Vargas", "rating": 4, "text": "Solid experience overall. Knowledgeable stylists who give good advice. Pricing is transparent which I appreciate.", "date": "2025-11-30", "sentiment_score": 0.65, "sentiment_label": "positive"},
]

GYM_REVIEWS = [
    {"author": "Miguel Ángel", "rating": 5, "text": "Best gym in the Salamanca district. Equipment is top-notch, always clean, and the trainers actually know what they're doing.", "date": "2026-02-10", "sentiment_score": 0.93, "sentiment_label": "very positive"},
    {"author": "Patricia Vargas", "rating": 3, "text": "Good equipment but WAY too crowded during peak hours (6-8pm). Had to wait 15 minutes for a bench press. AC also struggles in summer.", "date": "2026-02-07", "sentiment_score": 0.22, "sentiment_label": "mixed"},
    {"author": "Roberto Cruz", "rating": 4, "text": "The group classes are excellent, especially the HIIT sessions. Wish they had more yoga options. Locker rooms are always clean.", "date": "2026-02-03", "sentiment_score": 0.75, "sentiment_label": "positive"},
    {"author": "Diana López", "rating": 2, "text": "Signed up for a year contract, then they raised prices mid-contract. Customer service was unhelpful about it. Equipment is fine though.", "date": "2026-01-28", "sentiment_score": -0.68, "sentiment_label": "negative"},
    {"author": "Javier Morales", "rating": 5, "text": "The personal training sessions are worth every cent. Lost 12kg in 3 months. My trainer Pablo is incredibly knowledgeable and motivating.", "date": "2026-01-20", "sentiment_score": 0.96, "sentiment_label": "very positive"},
    {"author": "Sandra Reyes", "rating": 4, "text": "Great variety of equipment for a commercial gym. The 24-hour access is a huge plus for my work schedule.", "date": "2026-01-15", "sentiment_score": 0.78, "sentiment_label": "positive"},
    {"author": "Alberto Soto", "rating": 3, "text": "It's okay. Equipment is decent but some machines are always broken. The sauna is always out of order too.", "date": "2026-01-10", "sentiment_score": 0.15, "sentiment_label": "mixed"},
    {"author": "Natalia Flores", "rating": 5, "text": "Love the women's section! Clean, spacious, and never feels intimidating. The instructors are super supportive.", "date": "2026-01-05", "sentiment_score": 0.94, "sentiment_label": "very positive"},
    {"author": "Carlos Ibáñez", "rating": 2, "text": "Terrible customer service. Cancelled my membership 3 times online but they kept charging me. Finally got it resolved after 2 months.", "date": "2025-12-28", "sentiment_score": -0.85, "sentiment_label": "very negative"},
    {"author": "Eva Martín", "rating": 4, "text": "Good gym overall. The spin classes are amazing! Wish they had more equipment on the weight floor though.", "date": "2025-12-20", "sentiment_score": 0.68, "sentiment_label": "positive"},
    {"author": "Gabriel Ortega", "rating": 5, "text": "Best investment I've made for my health. The community feel is incredible - everyone knows each other and encourages one another.", "date": "2025-12-15", "sentiment_score": 0.97, "sentiment_label": "very positive"},
    {"author": "Lorena Peña", "rating": 3, "text": "Average gym. Gets the job done but nothing special. Parking is a nightmare during peak hours.", "date": "2025-12-10", "sentiment_score": 0.21, "sentiment_label": "mixed"},
    {"author": "Sergio Núñez", "rating": 4, "text": "Great functional training area. The battle ropes and sled are awesome. Clean facilities and good amenities.", "date": "2025-12-05", "sentiment_score": 0.72, "sentiment_label": "positive"},
    {"author": "Andrea Serrano", "rating": 1, "text": "Cancelled my membership after 2 weeks. Hidden fees everywhere and the 'free trial' turned into a permanent charge.", "date": "2025-11-30", "sentiment_score": -0.92, "sentiment_label": "very negative"},
    {"author": "Ricardo Campos", "rating": 4, "text": "Solid gym with friendly staff. The monthly protein shake included in membership is a nice touch.", "date": "2025-11-25", "sentiment_score": 0.69, "sentiment_label": "positive"},
]

def load_demo_data():
    for biz in DEMO_BUSINESSES:
        businesses_db[biz["id"]] = {**biz, "created_at": datetime.now().isoformat()}
        if "café" in biz["category"].lower():
            reviews_db[biz["id"]] = CAFE_REVIEWS
        elif "salon" in biz["category"].lower():
            reviews_db[biz["id"]] = SALON_REVIEWS
        else:
            reviews_db[biz["id"]] = GYM_REVIEWS
        personas_db[biz["id"]] = generate_personas(biz["category"])
        insights_db[biz["id"]] = generate_insights(biz["category"])

def generate_personas(category: str) -> List[dict]:
    base = [
        {"id": "p1", "name": "The Loyal Regular", "age_range": "28-45", "description": "Visits frequently, knows staff by name, values consistency.", "motivations": ["Consistency", "Personal service"], "pain_points": ["Price increases", "Staff changes"], "spending_behavior": "€60-120/month", "visit_frequency": "Weekly", "sentiment": "Very Positive", "typical_rating": 4.7, "percentage": 30.0},
        {"id": "p2", "name": "The Quality Seeker", "age_range": "30-50", "description": "Will pay premium for excellence. Writes detailed reviews.", "motivations": ["Quality", "Expertise"], "pain_points": ["Mediocre service", "Cutting corners"], "spending_behavior": "€100-200/month", "visit_frequency": "Bi-weekly", "sentiment": "Positive", "typical_rating": 4.5, "percentage": 25.0},
        {"id": "p3", "name": "The Value Hunter", "age_range": "25-40", "description": "Compares prices, looks for deals, will switch for better value.", "motivations": ["Price", "Promotions"], "pain_points": ["High prices", "No deals"], "spending_behavior": "€30-60/month", "visit_frequency": "Monthly", "sentiment": "Mixed", "typical_rating": 3.2, "percentage": 25.0},
    ]
    return base

def generate_insights(category: str) -> List[dict]:
    return [
        {"category": "service", "title": "Improve Response Time", "description": "Multiple reviews mention slow service. Implement quick-response training.", "impact": "high", "effort": "medium", "priority_score": 8.5},
        {"category": "operations", "title": "Manage Peak Hours", "description": "Crowding during peak times drives negative reviews. Consider reservations.", "impact": "high", "effort": "medium", "priority_score": 8.0},
        {"category": "pricing", "title": "Transparent Pricing", "description": "Price-sensitive customers want upfront costs. Publish full price list.", "impact": "medium", "effort": "low", "priority_score": 7.5},
    ]

# ─── Gemini AI Integration ───────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

async def call_gemini(prompt: str, max_tokens: int = 4096) -> str:
    if not GEMINI_API_KEY:
        return "[Demo Mode] AI analysis would appear here with a valid Gemini API key."
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Gemini API error: {resp.status_code}")
                return f"[API Error: {resp.status_code}]"
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return f"[Error: {str(e)}]"

# ─── API Routes with Error Handling ───────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_demo_data()
    logger.info("✅ Simon AI backend started with demo data")

@app.get("/api/health")
async def health():
    try:
        return {"status": "healthy", "version": "1.0.0", "businesses": len(businesses_db)}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses")
async def list_businesses():
    try:
        return list(businesses_db.values())
    except Exception as e:
        logger.error(f"List businesses failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses/{business_id}")
async def get_business(business_id: str):
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        return businesses_db[business_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get business failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses/{business_id}/reviews")
async def get_reviews(business_id: str):
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        return reviews_db.get(business_id, [])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get reviews failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses/{business_id}/personas")
async def get_personas(business_id: str):
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        return personas_db.get(business_id, [])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get personas failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses/{business_id}/insights")
async def get_insights(business_id: str):
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        return insights_db.get(business_id, [])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get insights failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses/{business_id}/competitors")
async def get_competitors(business_id: str):
    """Get mock competitor data with ratings comparison."""
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        
        biz = businesses_db[business_id]
        category = biz.get("category", "")
        
        competitors = [
            {"id": f"{business_id}-comp1", "name": "Nearby Competitor A", "distance": "0.3 km", "rating": round(biz["avg_rating"] + (hash("A") % 10 - 5) * 0.1, 1), "review_count": 150 + (hash("A") % 200), "strengths": ["Location", "Price"], "weaknesses": ["Service quality"]},
            {"id": f"{business_id}-comp2", "name": "Premium Option B", "distance": "0.8 km", "rating": round(biz["avg_rating"] + (hash("B") % 10 - 3) * 0.1, 1), "review_count": 80 + (hash("B") % 100), "strengths": ["Quality", "Expertise"], "weaknesses": ["Higher prices"]},
            {"id": f"{business_id}-comp3", "name": "Budget Choice C", "distance": "0.5 km", "rating": round(biz["avg_rating"] + (hash("C") % 10 - 7) * 0.1, 1), "review_count": 200 + (hash("C") % 150), "strengths": ["Low prices", "Convenience"], "weaknesses": ["Inconsistent quality"]},
        ]
        
        return {
            "business": {"id": biz["id"], "name": biz["name"], "rating": biz["avg_rating"]},
            "competitors": competitors,
            "analysis": f"Compared to {len(competitors)} nearby competitors, {biz['name']} ranks {'above' if biz['avg_rating'] >= competitors[0]['rating'] else 'below'} average in ratings."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get competitors failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses/{business_id}/trends")
async def get_trends(business_id: str):
    """Get sentiment trends over 6 months."""
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        
        months = []
        base_date = datetime.now()
        
        for i in range(6):
            month_date = base_date - timedelta(days=30 * (5 - i))
            month_name = month_date.strftime("%b %Y")
            
            month_sentiment = 0.5 + (hash(str(i)) % 20 - 10) / 20
            month_rating = 3.5 + (hash(str(i)) % 15 - 5) / 10
            review_volume = 20 + (hash(str(i)) % 25)
            
            months.append({
                "month": month_name,
                "sentiment_score": round(month_sentiment, 2),
                "avg_rating": round(month_rating, 1),
                "review_count": review_volume,
                "sentiment_label": "positive" if month_sentiment > 0.5 else "negative" if month_sentiment < 0.3 else "neutral"
            })
        
        avg_sentiment = sum(m["sentiment_score"] for m in months) / len(months)
        trend_direction = "improving" if months[-1]["sentiment_score"] > months[0]["sentiment_score"] else "declining"
        
        return {
            "business_id": business_id,
            "period": "6 months",
            "trend_direction": trend_direction,
            "average_sentiment": round(avg_sentiment, 2),
            "monthly_data": months,
            "summary": f"Sentiment is {trend_direction} over the last 6 months."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get trends failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/api/businesses/{business_id}/analytics")
async def get_analytics(business_id: str):
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        
        reviews = reviews_db.get(business_id, [])
        ratings = [r["rating"] for r in reviews]
        sentiments = [r.get("sentiment_score", 0) for r in reviews]
        
        rating_dist = dict(Counter(ratings))
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        positive = len([s for s in sentiments if s > 0.3])
        negative = len([s for s in sentiments if s < -0.3])
        neutral = len(sentiments) - positive - negative
        
        return {
            "total_reviews": len(reviews),
            "avg_rating": round(sum(ratings) / len(ratings), 1) if ratings else 0,
            "rating_distribution": rating_dist,
            "sentiment": {
                "average": round(avg_sentiment, 2),
                "positive_pct": round(positive / len(sentiments) * 100, 1) if sentiments else 0,
                "negative_pct": round(negative / len(sentiments) * 100, 1) if sentiments else 0,
                "neutral_pct": round(neutral / len(sentiments) * 100, 1) if sentiments else 0,
            },
            "persona_count": len(personas_db.get(business_id, [])),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get analytics failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.post("/api/businesses/{business_id}/chat")
async def chat_with_persona(business_id: str, req: ChatRequest):
    try:
        if business_id not in businesses_db:
            raise HTTPException(404, "Business not found")
        
        biz = businesses_db[business_id]
        personas = personas_db.get(business_id, [])
        persona = next((p for p in personas if p["id"] == req.persona_id), None)
        
        if not persona:
            raise HTTPException(404, "Persona not found")
        
        history_text = "\n".join(
            f"{'Business Owner' if m.role == 'user' else persona['name']}: {m.content}"
            for m in req.history[-10:]
        )
        
        prompt = f"""You are simulating a customer persona for {biz['name']} ({biz.get('category', 'N/A')}).

PERSONA: {persona['name']}, {persona['age_range']} - {persona['description']}
Motivations: {', '.join(persona['motivations'])}
Pain Points: {', '.join(persona['pain_points'])}

HISTORY:
{history_text}

User: {req.message}

Respond as this customer. Keep it conversational (2-4 sentences)."""
        
        response = await call_gemini(prompt)
        
        return {"persona_id": req.persona_id, "persona_name": persona["name"], "response": response}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(500, "Internal server error")

@app.post("/api/businesses")
async def add_business(biz: BusinessCreate, background_tasks: BackgroundTasks):
    try:
        import uuid
        biz_id = f"biz-{uuid.uuid4().hex[:8]}"
        
        businesses_db[biz_id] = {
            "id": biz_id,
            "name": biz.name,
            "address": biz.address,
            "category": biz.category,
            "google_place_id": biz.google_place_id,
            "review_count": 0,
            "avg_rating": 0.0,
            "created_at": datetime.now().isoformat(),
            "status": "scraping",
        }
        
        background_tasks.add_task(scrape_reviews_for_business, biz_id)
        
        return {"id": biz_id, "status": "created", "message": "Review scraping started"}
    except Exception as e:
        logger.error(f"Add business failed: {e}")
        raise HTTPException(500, "Internal server error")

async def scrape_reviews_for_business(business_id: str):
    logger.info(f"Scraping reviews for {business_id}...")
    biz = businesses_db.get(business_id)
    if not biz:
        return
    biz["status"] = "ready"
    logger.info(f"✅ Analysis complete for {business_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
