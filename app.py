#!/usr/bin/env python3
"""
Cambria AI - Live Call Transcription Monitor
No database, no persistence - calls exist only in memory
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from twilio.twiml.voice_response import VoiceResponse, Gather
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("Twilio not installed. Install with: pip install twilio")

# ====================== CONFIGURATION ======================
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    TWILIO_VOICE = os.getenv("TWILIO_VOICE", "Polly.Joanna")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    CLINIC_NAME = os.getenv("CLINIC_NAME", "Cambria Medical Center")
    RECEPTIONIST_NAME = os.getenv("RECEPTIONIST_NAME", "Sarah")
    CLINIC_HOURS = "Monday-Friday 8AM-5PM, Saturday 9AM-1PM"

config = Config()

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== IN-MEMORY STORAGE ======================

active_calls: Dict[str, Dict] = {}
# Structure: {
#   "call_sid": {
#     "session_id": "uuid",
#     "from": "+1234567890",
#     "to": "+0987654321",
#     "started_at": "2024-01-01T12:00:00",
#     "messages": [
#       {"role": "assistant", "content": "Hello...", "timestamp": "..."},
#       {"role": "user", "content": "Hi...", "timestamp": "..."}
#     ]
#   }
# }

# ====================== WEBSOCKET MANAGER ======================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to connection: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# ====================== AI PROCESSOR ======================

class AIProcessor:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
    
    async def process_message(self, message: str, conversation_history: List[Dict]) -> str:
        if not self.client:
            return "I apologize, but I'm currently unable to process your request."
        
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()}
            ]
            
            # Add recent conversation history (last 10 messages)
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"][:500]
                })
            
            messages.append({"role": "user", "content": message})
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=400
                ),
                timeout=30.0
            )
            
            return response.choices[0].message.content
            
        except asyncio.TimeoutError:
            return "I'm taking longer than expected. Please try again."
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return "I apologize, I'm having difficulty processing that. Could you please try again?"
    
    def _get_system_prompt(self) -> str:
        return f"""You are {config.RECEPTIONIST_NAME}, an AI medical receptionist at {config.CLINIC_NAME}.

This is a PHONE CALL. Keep responses brief and conversational.

PERSONALITY:
- Warm, empathetic, professional
- Patient and clear communicator
- Proactive in offering help

CLINIC INFORMATION:
- Hours: {config.CLINIC_HOURS}
- We accept most major insurance
- Same-day appointments available for urgent needs

GUIDELINES:
- Use natural conversational language
- Be helpful and informative
- Keep responses concise but complete
- Offer to schedule appointments when appropriate
- If asked about medical advice, politely explain you can only help with scheduling and general information

Respond naturally and helpfully to the caller's message."""

ai_processor = AIProcessor()

# ====================== FASTAPI APP ======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Cambria Live Call Transcription Monitor")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="Cambria Live Call Transcription",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== API ENDPOINTS ======================

@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Live Call Transcription Monitor"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "active_calls": len(active_calls),
        "websocket_connections": len(manager.active_connections)
    }

@app.get("/api/calls/active")
async def get_active_calls():
    return {
        "calls": list(active_calls.values()),
        "count": len(active_calls)
    }

# ====================== WEBSOCKET ======================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send current active calls
        await websocket.send_json({
            "type": "initial_state",
            "calls": list(active_calls.values())
        })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ====================== TWILIO WEBHOOKS ======================

@app.post("/twilio/voice")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio voice call"""
    if not TWILIO_AVAILABLE:
        return Response(content="Twilio not configured", status_code=503)
    
    try:
        form = await request.form()
        call_sid = form.get('CallSid')
        from_number = form.get('From')
        to_number = form.get('To')
        
        logger.info(f"📞 Incoming call from {from_number}, CallSid: {call_sid}")
        
        # Create call record in memory
        session_id = call_sid  # Use call_sid as session_id for simplicity
        active_calls[call_sid] = {
            "session_id": session_id,
            "call_sid": call_sid,
            "from": from_number,
            "to": to_number,
            "started_at": datetime.utcnow().isoformat(),
            "messages": []
        }
        
        # Greeting
        greeting = f"Hello! Welcome to {config.CLINIC_NAME}. I'm {config.RECEPTIONIST_NAME}, your A I assistant. How can I help you today?"
        
        # Add greeting to messages
        active_calls[call_sid]["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Broadcast call start
        await manager.broadcast({
            "type": "call_started",
            "call": active_calls[call_sid]
        })
        
        # Create TwiML response
        response = VoiceResponse()
        response.say(greeting, voice=config.TWILIO_VOICE)
        
        gather = Gather(
            input='speech',
            action=f"/twilio/process/{session_id}",
            method="POST",
            timeout=5,
            speech_timeout='auto',
            language='en-US'
        )
        response.append(gather)
        
        response.say("I didn't hear anything. Goodbye!", voice=config.TWILIO_VOICE)
        response.hangup()
        
        return Response(content=str(response), media_type="text/xml")
        
    except Exception as e:
        logger.error(f"Failed to handle call: {e}", exc_info=True)
        response = VoiceResponse()
        response.say("I'm sorry, there's a technical problem. Please try again later.", 
                     voice=config.TWILIO_VOICE)
        response.hangup()
        return Response(content=str(response), media_type="text/xml")


@app.post("/twilio/process/{session_id}")
async def process_twilio_speech(session_id: str, request: Request):
    """Process speech input from Twilio"""
    try:
        form = await request.form()
        speech_result = form.get('SpeechResult', '')
        call_sid = form.get('CallSid')
        
        logger.info(f"🎤 Speech from {call_sid}: '{speech_result}'")
        
        # Find call by call_sid or session_id
        call_data = active_calls.get(call_sid) or active_calls.get(session_id)
        
        if not call_data:
            logger.error(f"Call {call_sid} not found")
            response = VoiceResponse()
            response.say("Session not found. Goodbye!", voice=config.TWILIO_VOICE)
            response.hangup()
            return Response(content=str(response), media_type="text/xml")
        
        response = VoiceResponse()
        
        if speech_result and len(speech_result.strip()) > 0:
            # Add user message
            call_data["messages"].append({
                "role": "user",
                "content": speech_result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Broadcast user message
            await manager.broadcast({
                "type": "message_added",
                "call_sid": call_data["call_sid"],
                "message": call_data["messages"][-1]
            })
            
            # Process with AI
            try:
                ai_response = await ai_processor.process_message(
                    speech_result, 
                    call_data["messages"]
                )
                
                # Add AI message
                call_data["messages"].append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Broadcast AI message
                await manager.broadcast({
                    "type": "message_added",
                    "call_sid": call_data["call_sid"],
                    "message": call_data["messages"][-1]
                })
                
                # Say the response
                response.say(ai_response, voice=config.TWILIO_VOICE)
                
            except Exception as e:
                logger.error(f"AI error: {e}")
                ai_response = "I'm having trouble processing that right now."
                response.say(ai_response, voice=config.TWILIO_VOICE)
            
            # Check if conversation should end
            goodbye_phrases = [
                'no thank', 'that\'s all', 'goodbye', 'bye', 
                'thank you', 'thanks', 'nothing else', 'all set'
            ]
            
            should_end = any(phrase in speech_result.lower() for phrase in goodbye_phrases)
            ai_farewell = any(phrase in ai_response.lower() for phrase in ['have a great day', 'goodbye', 'take care'])
            
            if should_end or (ai_farewell and not speech_result.endswith('?')):
                response.say("Thank you for calling. Goodbye!", voice=config.TWILIO_VOICE)
                response.hangup()
            else:
                gather = Gather(
                    input='speech',
                    action=f"/twilio/process/{session_id}",
                    method="POST",
                    timeout=5,
                    speech_timeout='auto',
                    language='en-US'
                )
                
                if not ai_response.strip().endswith('?'):
                    gather.say("Is there anything else I can help you with?", voice=config.TWILIO_VOICE)
                
                response.append(gather)
                response.say("Thank you for calling. Goodbye!", voice=config.TWILIO_VOICE)
                response.hangup()
            
        else:
            logger.warning(f"No speech detected for {call_sid}")
            response.say("I didn't hear anything. Goodbye!", voice=config.TWILIO_VOICE)
            response.hangup()
        
        return Response(content=str(response), media_type="text/xml")
        
    except Exception as e:
        logger.error(f"Failed to process speech: {e}", exc_info=True)
        response = VoiceResponse()
        response.say("I'm sorry, I had a technical problem. Please try calling again.", 
                     voice=config.TWILIO_VOICE)
        response.hangup()
        return Response(content=str(response), media_type="text/xml")


@app.post("/twilio/voice/status")
async def handle_call_status(request: Request):
    """Handle Twilio call status updates"""
    try:
        form = await request.form()
        call_sid = form.get('CallSid')
        call_status = form.get('CallStatus')
        
        logger.info(f"📊 Call status: {call_sid} -> {call_status}")
        
        # If call ended, remove from active calls
        if call_status in ['completed', 'failed', 'busy', 'no-answer', 'canceled']:
            if call_sid in active_calls:
                # Broadcast call ended
                await manager.broadcast({
                    "type": "call_ended",
                    "call_sid": call_sid
                })
                
                # Remove from memory
                del active_calls[call_sid]
                logger.info(f"Call {call_sid} removed from memory")
        
        return Response(content="OK", media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Call status error: {e}")
        return Response(content="OK", media_type="text/plain")


# ====================== MAIN ======================

if __name__ == "__main__":
    logger.info(f"Starting Cambria Live Call Monitor on port {config.PORT}")
    logger.info(f"Twilio: {'✓' if TWILIO_AVAILABLE else '✗'}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=config.DEBUG,
        log_level="debug" if config.DEBUG else "info"
    )