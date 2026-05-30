# app/services/agent_service.py
 
import os
import json
import base64
import cv2
from datetime import datetime  
from app.core.config import settings
from app.core.prompts import VISION_EXTRACTION_PROMPT, FINAL_REPORT_PROMPT
from app.services.rag_service import RAGService
from app.core.graph_logic import TrafficLogicGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
 
class AccidentAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.VLM_MODEL,
            max_tokens=1500,
            temperature=0.1
        )
        self.chat_history = []
        self.rag = RAGService()
        self.logic_graph = TrafficLogicGraph()
 
    def encode_image(self, image_path, max_width=512):
        img = cv2.imread(image_path)
        if img is None:
            return ""
        height, width = img.shape[:2]
        if width > max_width:
            ratio = max_width / width
            new_size = (max_width, int(height * ratio))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return ""
        return base64.b64encode(buffer).decode('utf-8')
 
    def generate_initial_analysis(self, keyframes):
        # Generate the exact time the system started processing the crash
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
        # --- PHASE 1: Vision to JSON ---
        vision_content = [{"type": "text", "text": VISION_EXTRACTION_PROMPT}]
        for path in keyframes:
            if os.path.exists(path):
                vision_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(path)}"}})
       
        print("[System] Phase 1: Extracting visual states and descriptions...")
        json_response = self.llm.invoke([HumanMessage(content=vision_content)]).content
       
        try:
            clean_json = json_response.replace('```json', '').replace('```', '').strip()
            visual_data = json.loads(clean_json)
           
            # Extract both description and state for each vehicle
            veh_a = visual_data.get("Vehicle_A", {})
            veh_b = visual_data.get("Vehicle_B", {})
           
            desc_a = veh_a.get("description", "first vehicle")
            state_a = veh_a.get("state", "Main Road")
           
            desc_b = veh_b.get("description", "second vehicle")
            state_b = veh_b.get("state", "Side Road")
           
        except Exception as e:
            print(f"[Warning] JSON Parsing failed: {e}")
            desc_a, state_a = "first vehicle", "Main Road"
            desc_b, state_b = "second vehicle", "Side Road"
 
        # --- PHASE 2: Graph Logic Evaluation ---
        print(f"[System] Phase 2: Graph Evaluating A({state_a}) vs B({state_b})...")
        graph_verdict = self.logic_graph.determine_liability(state_a, state_b)
 
        # --- PHASE 3: FAISS Retrieval & Final Report ---
        print("[System] Phase 3: Retrieving PSD Manual rules...")
        legal_context = self.rag.get_relevant_law(query=f"rules regarding {state_a} and {state_b}")
       
        final_system_prompt = FINAL_REPORT_PROMPT.format(
            timestamp=current_timestamp,
            desc_a=desc_a,
            state_a=state_a,
            desc_b=desc_b,
            state_b=state_b,
            graph_verdict=graph_verdict,
            legal_context=legal_context
        )
       
        self.chat_history = [SystemMessage(content=final_system_prompt)]
        self.chat_history.append(HumanMessage(content="Generate the final liability report based on the provided data."))
       
        print("[System] Generating deterministic legal report...")
        final_report = self.llm.invoke(self.chat_history)
        self.chat_history.append(AIMessage(content=final_report.content))
       
        return final_report.content
 
    def chat_with_user(self, user_message):
        self.chat_history.append(HumanMessage(content=user_message))
        response = self.llm.invoke(self.chat_history)
        self.chat_history.append(AIMessage(content=response.content))
<<<<<<< HEAD
        return response.content
=======
        return response.content
    
>>>>>>> dcd2c80127c3cebb58a29cfe7eb6913f565d56d6
