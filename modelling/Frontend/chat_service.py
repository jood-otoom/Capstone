import gradio as gr
from html import escape
from ui_app.agent_service import safe_agent_call

def render_chat_html(chat_history):
    if not chat_history:
        return """
        <div style="text-align: center; color: var(--text-muted); padding: 40px 20px; font-size: 0.98rem; line-height: 1.6; background: rgba(248, 250, 252, 0.6); border: 1px dashed var(--card-border); border-radius: 20px; margin: 18px 0; min-height: 480px; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 10px;">
            <div style="font-size: 2.2rem;">&#129302;</div>
            <div style="font-weight: 800; color: var(--primary-dark); font-size: 1.15rem;">Accident Law Assistant Chat</div>
            <div style="max-width: 420px; opacity: 0.85;">
                Awaiting inference result. Upload a road scene or video and run detection to automatically initiate the AI Accident Liability Analysis.
            </div>
        </div>
        """
    
    html = '<div class="chat-preview" style="height: 480px; max-height: 480px; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; margin: 18px 0; background: rgba(248, 250, 252, 0.3); border-radius: 16px; border: 1px solid var(--card-border);">'
    for role, text in chat_history:
        if role == "User":
            html += f"""
            <div class="chat-bubble chat-bubble-user" style="align-self: flex-end; background: #2563eb !important; border: 1px solid #1e40af; color: #ffffff !important; max-width: 85%; padding: 12px 16px; border-radius: 18px 18px 2px 18px; font-size: 0.92rem; line-height: 1.5; text-align: left; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);">
                <span class="chat-role" style="display: block; margin-bottom: 4px; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: #dbeafe !important;">User</span>
                <div style="word-break: break-word; white-space: pre-wrap; color: #ffffff !important;">{escape(text)}</div>
            </div>
            """
        else:
            # Safely render the text and preserve formatting / line breaks
            formatted_text = escape(text).replace('\\n', '<br>').replace('\n', '<br>')
            # Support basic formatting like bold or bullet points if any
            # Format bold blocks safely
            parts = formatted_text.split('**')
            new_text = ""
            for idx, part in enumerate(parts):
                if idx % 2 == 1:
                    new_text += f"<b>{part}</b>"
                else:
                    new_text += part
            formatted_text = new_text
            
            # Replaces * bullet points with bullet symbol
            formatted_text = formatted_text.replace('<br>* ', '<br>&bull; ')
            formatted_text = formatted_text.replace('<br>- ', '<br>&bull; ')
            
            html += f"""
            <div class="chat-bubble chat-bubble-assistant" style="align-self: flex-start; background: #ffffff; border: 1px solid rgba(147, 197, 253, 0.45); color: #0f172a; max-width: 85%; padding: 12px 16px; border-radius: 18px 18px 18px 2px; font-size: 0.92rem; line-height: 1.5; box-shadow: 0 4px 12px rgba(29, 78, 216, 0.04); text-align: left;">
                <span class="chat-role" style="display: block; margin-bottom: 4px; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: #1e40af;">Assistant</span>
                <div style="word-break: break-word; white-space: pre-wrap; color: #0f172a !important;">{formatted_text}</div>
            </div>
            """
    html += '</div>'
    return html

def initiate_chat(user_message, chat_history):
    chat_history = chat_history or []
    if not user_message or not user_message.strip():
        return "", render_chat_html(chat_history), chat_history, gr.update(), gr.update()
    chat_history.append(("User", user_message))
    chat_history.append(("Assistant", "⏳ Thinking..."))
    return (
        "",
        render_chat_html(chat_history),
        chat_history,
        gr.update(interactive=False, placeholder="Analyzing query with AI agent..."),
        gr.update(interactive=False, value="Sending...")
    )

def generate_chat_reply(chat_history, agent):
    if not chat_history:
        return render_chat_html([]), [], gr.update(interactive=True, placeholder="Ask the legal assistant..."), gr.update(interactive=True, value="Send")
    
    user_message = ""
    if len(chat_history) >= 2:
        user_role, user_message = chat_history[-2]
    elif len(chat_history) == 1:
        user_role, user_message = chat_history[-1]
    
    if not agent:
        chat_history[-1] = ("Assistant", "⚠️ AI Agent is offline. OpenRouter key or required files are missing.")
        return render_chat_html(chat_history), chat_history, gr.update(interactive=True, placeholder="Ask the legal assistant..."), gr.update(interactive=True, value="Send")

    try:
        reply = safe_agent_call(agent, "chat_with_user", user_message)
        chat_history[-1] = ("Assistant", reply)
    except Exception as e:
        import traceback
        print(f"[Chat Error]: {e}")
        traceback.print_exc()
        chat_history[-1] = ("Assistant", f"❌ Agent analysis failed: {str(e)}")

    return render_chat_html(chat_history), chat_history, gr.update(interactive=True, placeholder="Ask the legal assistant..."), gr.update(interactive=True, value="Send")
