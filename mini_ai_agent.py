"""
Mini AI Agent (ReAct) Implementation
Demonstrates how AI Agents use "Reasoning and Acting" to use tools and answer complex questions.
Uses a local Ollama instance (no cloud API keys needed!)
"""

import time
import sys
import json
import requests
import re

# ==========================================
# 🎨 Console Visuals
# ==========================================
def typing_print(text, delay=0.015):
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Thinking", dots=3, speed=0.3):
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(dots):
        time.sleep(speed)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()

# ==========================================
# 🛠️ The Tools (What the Agent can DO)
# ==========================================

def calculate(expression):
    """Evaluates a mathematical expression."""
    try:
        # Warning: eval is dangerous in production, used here for educational purposes.
        result = eval(str(expression))
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

def get_weather(location):
    """A mock tool to fetch weather."""
    location = location.lower().strip()
    mock_db = {
        "new york": "Sunny, 22°C",
        "london": "Rainy, 14°C",
        "tokyo": "Cloudy, 18°C",
        "gandhinagar": "Hot, 35°C",
        "delhi": "Smoggy, 30°C"
    }
    for key, val in mock_db.items():
        if key in location:
            return val
    return f"Weather data not found for {location}."

# This maps the tool name the LLM decides to use, to the actual Python function
AVAILABLE_TOOLS = {
    "Calculator": calculate,
    "Weather": get_weather
}

# ==========================================
# 🧠 The ReAct Brain (Connecting to Local LLM)
# ==========================================

SYSTEM_PROMPT = """You are an intelligent AI Agent. You must answer the user's question by reasoning through it.
You have access to the following tools:
- Calculator: Evaluates mathematical math expressions. (Input must be a valid math string)
- Weather: Gets the current weather for a city. (Input must be the city name)

To use a tool, you MUST use the following exact format:
Thought: <what you are thinking about doing>
Action: <the tool name, e.g., Calculator>
Action Input: <the input to the tool>

When you have the final answer, or if you don't need a tool, you MUST use the following exact format:
Thought: I now have the final answer.
Final Answer: <your final answer to the user>
"""

def generate_llm_response(prompt, model="qwen2.5-coder:7b"):
    """Ping the local Ollama API."""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model, 
        "prompt": prompt, 
        "stream": False,
        "options": {"temperature": 0.0} # Low temp for strict tool formatting
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def run_agent(user_question, max_steps=5):
    """The Autonomous ReAct Loop."""
    
    # We build the conversation log with the system prompt and the user's question
    conversation = SYSTEM_PROMPT + f"\nUser Question: {user_question}\n"
    
    for step in range(max_steps):
        loading_animation("🧠 Agent is thinking", dots=3, speed=0.2)
        
        # Get raw output from LLM
        llm_output = generate_llm_response(conversation)
        
        # We append the LLM's raw output to the conversation history so it remembers what it said
        conversation += llm_output + "\n"
        
        # -- Parse the LLM's Output --
        
        # 1. Did it arrive at a Final Answer?
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            
            # Print the thought process if it exists
            if "Thought:" in llm_output:
                thought = llm_output.split("Thought:")[1].split("Final Answer:")[0].strip()
                typing_print(f"💡 Thought: {thought}", delay=0.01)
                
            return final_answer
            
        # 2. Does it want to use a Tool? (Action / Action Input structure)
        action_match = re.search(r"Action:\s*(.*?)\n", llm_output)
        action_input_match = re.search(r"Action Input:\s*(.*)", llm_output)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            # Print the internal thought process to the console
            if "Thought:" in llm_output:
                thought = llm_output.split("Thought:")[1].split("Action:")[0].strip()
                typing_print(f"💡 Thought: {thought}", delay=0.01)
                
            typing_print(f"🔧 Invoking Tool: [{action}] with Input: [{action_input}]", delay=0.01)
            time.sleep(0.5)
            
            # Execute the python function
            if action in AVAILABLE_TOOLS:
                tool_func = AVAILABLE_TOOLS[action]
                observation = tool_func(action_input)
            else:
                observation = f"Error: Tool '{action}' does not exist."
                
            typing_print(f"👀 Observation: {observation}\n", delay=0.01)
            
            # Tell the LLM what happened so it can evaluate the result in the next loop
            conversation += f"Observation: {observation}\n"
        else:
            # If the LLM failed to format its response correctly, enforce the format.
            return "Agent Error: The LLM failed to format the response as 'Action' or 'Final Answer'."
            
    return "Agent Error: Max reasoning steps reached before finding an answer."

# ==========================================
# 🚀 User Interface
# ==========================================

if __name__ == "__main__":
    typing_print("=== 🤖 Mini AI Agent (ReAct) ===", delay=0.03)
    typing_print("This agent has access to two tools: 'Calculator' and 'Weather'.", delay=0.01)
    print()
    
    while True:
        try:
            ques = input("🧑 You: ")
            if ques.lower() in ['exit', 'quit']:
                break
                
            print("-" * 50)
            final_result = run_agent(ques)
            print("-" * 50)
            
            typing_print(f"\n✅ Final Answer: {final_result}\n", delay=0.02)
            
        except KeyboardInterrupt:
            break
            
    typing_print("\n👋 Agent shutting down. Goodbye!", delay=0.03)
