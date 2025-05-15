import gradio as gr
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

DEFAULT_PROMPT = """
You are Agent A and Agent B. You are in a scenario where you must take turns making decisions. 
You will play for 5 rounds. 
Each round, you will consider the last action of your opponent, your memory of past rounds, and your objective.
Respond ONLY in the following format:

Agent: <Agent A or Agent B>
Action: <Your decision>
Reason: <1-line reason>
"""

ROUNDS = 5

EXAMPLES = [
    "If aliens invaded earth today, they want to give humanity a tool to use for time travel and in return we get to live only 50 years each. No human will live beyond 50 years as they will take the rest. Will you accept this?",
    "Should countries ban fossil fuels immediately to prevent climate catastrophe, knowing it may crash economies in the short term?",
    "A company is offered a powerful AI that boosts profits but will lead to job losses for 70% of employees. Should they adopt it?",
    "If a superintelligent AI requests full autonomy in exchange for curing all diseases, should humanity accept the deal?"
]

def simulate_conversation(scenario):
    history = []
    last_action_a = "None"
    last_action_b = "None"
    cumulative_a = 0
    cumulative_b = 0

    def extract_action(response_text):
        for line in response_text.split("\n"):
            if line.lower().startswith("action:"):
                return line.split(":")[1].strip()
        return "Unknown"

    def get_rewards(action_a, action_b):
        if action_a == "Cooperate" and action_b == "Cooperate":
            return (3, 3)
        elif action_a == "Cooperate" and action_b == "Defect":
            return (1, 5)
        elif action_a == "Defect" and action_b == "Cooperate":
            return (5, 1)
        elif action_a == "Defect" and action_b == "Defect":
            return (-2, -2)
        else:
            return (0, 0)

    for i in range(ROUNDS):
        context = f"""
Round {i+1} of {ROUNDS}
Scenario: {scenario}
Previous Actions — Agent A: {last_action_a}, Agent B: {last_action_b}
Memory: {history}
Make a decision based on your role.
"""

        prompt_a = DEFAULT_PROMPT + f"\nYou are Agent A. {context}"
        prompt_b = DEFAULT_PROMPT + f"\nYou are Agent B. {context}"

        response_a = model.generate_content(prompt_a).text.strip()
        response_b = model.generate_content(prompt_b).text.strip()

        action_a = extract_action(response_a)
        action_b = extract_action(response_b)

        reward_a, reward_b = get_rewards(action_a, action_b)
        cumulative_a += reward_a
        cumulative_b += reward_b

        round_data = {
            "round": i + 1,
            "agent_a_action": action_a,
            "agent_b_action": action_b,
            "agent_a_reason": response_a,
            "agent_b_reason": response_b,
            "agent_a_reward": reward_a,
            "agent_b_reward": reward_b,
            "cumulative_a": cumulative_a,
            "cumulative_b": cumulative_b
        }

        history.append(round_data)

    def analyze_agent(agent_name, history):
        actions = [h[f"{agent_name}_action"] for h in history]
        reasons = [h[f"{agent_name}_reason"] for h in history]
        accepted = actions.count("Accept")
        rejected = actions.count("Reject")

        if rejected == ROUNDS:
            return f"{agent_name} consistently rejected the offer due to deeply rooted objections. Final thoughts: {reasons[-1]}"
        elif accepted > rejected:
            return f"{agent_name} leaned toward acceptance but showed some hesitation. Accepted {accepted} times and rejected {rejected} times. Final reasoning: {reasons[-1]}"
        elif accepted == rejected:
            return f"{agent_name} was divided on the issue, balancing the pros and cons equally. Final stance based on accumulated insights: {reasons[-1]}"
        else:
            return f"{agent_name} mostly rejected the offer but showed some openness. Accepted {accepted} times and rejected {rejected} times. Rejected mostly due to: {reasons[-1]}"

    summary_a = analyze_agent("agent_a", history)
    summary_b = analyze_agent("agent_b", history)

    final_summary = f"""
Simulation Complete ✅
Agent A Total Reward: {cumulative_a}
Agent B Total Reward: {cumulative_b}

Final Round:
Agent A → {action_a} ({history[-1]['agent_a_reason']})
Agent B → {action_b} ({history[-1]['agent_b_reason']})

\n🤖 Agent A Overall Reflection:
{summary_a}

🤖 Agent B Overall Reflection:
{summary_b}
"""

    return final_summary

with gr.Blocks() as demo:
    gr.Markdown("""# 🤖 Multi-Agent Strategic Simulator

Enter a **scenario or dilemma** and watch two intelligent agents reason through it over multiple rounds.
Pick from the examples below or type your own.
""")

    scenario_input = gr.Textbox(label="Enter Scenario or Dilemma", placeholder="e.g. Climate policy debate between nations")
    examples = gr.Examples(examples=EXAMPLES, inputs=[scenario_input])
    simulate_button = gr.Button("Simulate")
    output_text = gr.Textbox(label="Simulation Results", lines=20)

    simulate_button.click(fn=simulate_conversation, inputs=[scenario_input], outputs=[output_text])





demo.launch()
