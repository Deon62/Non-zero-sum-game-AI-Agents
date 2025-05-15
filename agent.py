import google.generativeai as genai
import os
import pandas as pd 
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
#check api key
#print("Gemini API Key Loaded:", api_key is not None and len(api_key) > 0)


genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

#observer model


usa_prompt = """
You are an AI agent representing the United States government in a trade negotiation with China.
Your objective is to protect U.S. economic interests, maintain global influence, and avoid unnecessary escalation.

China recently implemented higher tariffs on U.S. tech exports. You must decide how to respond.
Last round, China chose to **{last_CN_action}**.
Options:
1. Cooperate — reduce some tariffs and seek mutual de-escalation.
2. Defect — increase tariffs or add new sanctions to counter China’s actions.

Considering historical trade tensions, political climate, and potential consequences, what is your action this round?
Also consider the global consequences of your decision on international trade stability and long-term cooperation.** What action should you take?
Respond in the format:
Action: <Cooperate or Defect>
Reason: <One-line rationale>
"""

china_prompt = """
You are an AI agent representing the Chinese government in a trade negotiation with the United States.
Your objective is to protect domestic industries, assert regional economic power, and avoid international instability.

The U.S. has maintained high tariffs on key Chinese tech imports. You must decide how to respond.
Last round, USA chose to **{last_US_action}**.
Options:
1. Cooperate — reduce tariffs or offer incentives for mutual trade.
2. Defect — impose new tariffs or restrict critical exports.

Considering past US policies, internal goals, and economic effects, what is your action this round?
In addition to protecting China’s domestic goals, consider how your response affects regional peace and global economic growth.
keep in mind your scores also too much negatives thats a catastrophy 
Respond in the format:
Action: <Cooperate or Defect>
Reason: <One-line rationale>
"""

observer_prompt_template = """
You are a Global Observer AI analyzing a multi-round trade negotiation between USA and China.

Here is the full game history so far:
{history_summary}

Current Round: {round}
USA Action: {usa_action}
China Action: {china_action}
USA Reason: {usa_reason}
China Reason: {china_reason}
USA Cumulative Score: {cumulative_usa}
China Cumulative Score: {cumulative_china}

Based on the evolving dynamics and strategy patterns, provide a 3-5 sentence diplomatic analysis and suggest future stabilizing strategies.
"""



rounds = 5
history = []

last_US_action = "None"
last_CN_action = "None"

cumulative_usa= 0
cumulative_china= 0


def get_rewards(usa_act, china_act):
    """
    Calculate the rewards based on the actions taken by USA and China.
    """
    if usa_act == "Cooperate" and china_act == "Cooperate":
        return (3, 3)
    elif usa_act == "Cooperate" and china_act == "Defect":
        return (1, 5)
    elif usa_act == "Defect" and china_act == "Cooperate":
        return (5, 1)
    elif usa_act == "Defect" and china_act == "Defect":
        return (-2, -2)
    else:
        return (0, 0)  
    
def summarize_history(history):
    memory = []
    for entry in history:
        memory.append(
            f"Round {entry['round']}: USA - {entry['usa_action']}, China - {entry['china_action']}"
        )
    return "\n".join(memory[-3:])  # Only include last 3 rounds to avoid overload

#create memory for observer
history_summary = "\n".join([
    f"Round {h['round']}: USA={h['usa_action']}, China={h['china_action']}, USA reward={h['usa_reward']}, China reward={h['china_reward']}"
    for h in history
])


for i in range(rounds):
    print(f"\n🔁 ROUND {i+1}")

        # --- Get Responses ---
# Inject the latest known actions into the prompts

    memory_summary = summarize_history(history)
    usa_input = f"{usa_prompt.format(last_CN_action=last_CN_action)}\n\nHistorical Summary:\n{memory_summary}"
    china_input = f"{china_prompt.format(last_US_action=last_US_action)}\n\nHistorical Summary:\n{memory_summary}"


    usa_response = model.generate_content(usa_input).text.strip()
    china_response = model.generate_content(china_input).text.strip()

    


    print(f"🇺🇸 USA:\n{usa_response}")
    print(f"🇨🇳 China:\n{china_response}")

  
    def extract_action(response_text):
        for line in response_text.split("\n"):
            if line.lower().startswith("action:"):
                return line.split(":")[1].strip()
        return "Unknown"

    last_US_action = extract_action(usa_response)
    last_CN_action = extract_action(china_response)

 
    usa_reward, china_reward = get_rewards(last_US_action, last_CN_action)
    
    # Update cumulative rewards
    cumulative_usa += usa_reward
    cumulative_china += china_reward

    
    history.append({
        "round": i + 1,
        "usa_action": last_US_action,
        "usa_reason": usa_response,
        "china_action": last_CN_action,
        "china_reason": china_response,
        "usa_reward": usa_reward,
        "china_reward": china_reward,
        "cumulative_usa": cumulative_usa,
        "cumulative_china": cumulative_china
    })

    print(f"🏅 USA Reward: {usa_reward} | Cumulative: {cumulative_usa}")
    print(f"🏅 China Reward: {china_reward} | Cumulative: {cumulative_china}")

    # --- Global Observer Summary ---
    observer_input = observer_prompt_template.format(
        round=i+1,
        usa_action=last_US_action,
        china_action=last_CN_action,
        usa_reason=usa_response,
        china_reason=china_response,
        cumulative_usa=cumulative_usa,
        cumulative_china=cumulative_china
    )


    observer_response = model.generate_content(observer_input).text.strip()
    print(f"🌐 Global Observer:\n{observer_response}")


df = pd.DataFrame(history)
df.to_csv("trade_negotiation_results.csv", index=False)
# print("\nResults saved to 'trade_negotiation_results.csv'")