import os
import asyncio
import argparse
from openai import AsyncOpenAI
from typing import List, Dict, Any


class MetaCognitiveAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def call_openai(self, system_role: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo", temperature=0.6, top_p=0.7,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    async def ask_question(self, question: str, context: str) -> str:
        system_role = "You are an expert in analyzing and answering questions about given contexts."
        user_prompt = f"""Context: {context}

Question: {question}

Provide an accurate and concise answer."""

        return await self.call_openai(system_role, user_prompt)

    async def process_questions(self, questions: List[str], input_data: str) -> Dict[str, str]:
        tasks = [self.ask_question(question, input_data) for question in questions]
        answers = await asyncio.gather(*tasks)
        return dict(zip(questions, answers))

    async def generate_class_labels(self, analysis: Dict[str, str]) -> List[str]:
        system_role = "You are an expert in categorizing and labeling data."
        user_prompt = f"""Based on the following analysis, generate a list of accurate value labels:

Analysis: {analysis}

Provide your answer as a comma-separated list of labels.
Output only your answer as a comma-separated list of labels."""

        response = await self.call_openai(system_role, user_prompt)
        return [label.strip() for label in response.split(',')]

    async def choose_template(self, analysis: Dict[str, str], class_labels: List[str]) -> str:
        system_role = "You are an expert in selecting appropriate reasoning templates."
        user_prompt = f"""Based on the following analysis and class labels, choose the most appropriate reasoning template:

Analysis: {analysis}
Class Labels: {class_labels}

Templates:
1. Deductive Reasoning: "If A is true, and B is true, then C must be true."
2. Inductive Reasoning: "Based on observations X, Y, and Z, we can generalize that..."
3. Abductive Reasoning: "The best explanation for phenomenon P is hypothesis H because..."
4. Analogical Reasoning: "Situation A is similar to situation B in ways X and Y, so we can infer..."

Provide your answer as the number of the chosen template. Output the number only."""

        response = await self.call_openai(system_role, user_prompt)
        templates = {
            "1": "If {premise1} is true, and {premise2} is true, then {conclusion} must be true.",
            "2": "Based on observations {observation1}, {observation2}, and {observation3}, we can generalize that {generalization}.",
            "3": "The best explanation for {phenomenon} is {hypothesis} because {reasoning}.",
            "4": "Situation {situationA} is similar to situation {situationB} in ways {similarity1} and {similarity2}, so we can infer {inference}."
        }
        
        return templates.get(response.strip(), templates["1"])  # Default to template 1 if invalid response

    async def fill_template(self, template: str, analysis: Dict[str, str], class_labels: List[str]) -> str:
        system_role = "You are an expert in applying reasoning strategies and filling in templates."
        user_prompt = f"""Fill in the following template based on the given analysis and class labels:

Template: {template}
Analysis: {analysis}
Class Labels: {class_labels}

Provide your answer as the completed template."""

        return await self.call_openai(system_role, user_prompt)

    async def final_completion(self, filled_template: str, analysis: Dict[str, str], class_labels: List[str]) -> str:
        system_role = "You are a meta-cognitive reasoning expert capable of synthesizing information and providing insightful conclusions."
        user_prompt = f"""Based on the following filled template, analysis, and class labels, provide a final completion that includes:
1. A summary of the reasoning process
2. An evaluation of the strength of the conclusion
3. Suggestions for further investigation or alternative perspectives

Context: {filled_template}
Analysis: {analysis}
Class Labels: {class_labels}

Provide your answer in a clear, structured format."""

        return await self.call_openai(system_role, user_prompt)

    async def final_answer(self, input_data: str, filled_template: str) -> str:
        system_role = "You are a meta-cognitive reasoning expert capable of synthesizing information and providing insightful conclusions."
        user_prompt = f"""Below is a conversation between a user and a helpful assistant.  Generate an accurate completion based on the context.

Context: {filled_template}
Input: {input_data}
Output:"""

        return await self.call_openai(system_role, user_prompt)

    
    async def meta_reasoning_loop(self, input_data: str) -> Dict[str, Any]:
        questions = [
            "What is the main topic or subject?",
            "What is the complexity level (low, medium, high)?",
            "What are the key concepts or ideas mentioned?",
            "What type of reasoning would be most appropriate (deductive, inductive, abductive, analogical)?",
            "Are there any apparent biases or logical fallacies?",
            "What additional information might be needed?"
            "What is the format of the reply being requested?"            
        ]

        analysis = await self.process_questions(questions, input_data)
        class_labels = await self.generate_class_labels(analysis)
        template = await self.choose_template(analysis, class_labels)
        filled_template = await self.fill_template(template, analysis, class_labels)
        final_output = await self.final_completion(filled_template, analysis, class_labels)
        final_answer = await self.final_answer(input_data,filled_template)
        return {
            "input_data": input_data,
            "analysis": analysis,
            "class_labels": class_labels,
            "chosen_template": template,
            "filled_template": filled_template,
            "final_output": final_output,
            "final_answer": final_answer            
        }

async def main():
    parser = argparse.ArgumentParser(description="MetaCognitive Agent with CLI input")
    parser.add_argument("--input", type=str, help="Input string or path to input file")
    parser.add_argument("--context", type=str, help="Additional context string or path to context file")
    args = parser.parse_args()

    # Read input
    if os.path.isfile(args.input):
        with open(args.input, 'r') as f:
            input_data = f.read()
    else:
        input_data = args.input

    # Read context if provided
    context = ""
    if args.context:
        if os.path.isfile(args.context):
            with open(args.context, 'r') as f:
                context = f.read()
        else:
            context = args.context

    # Combine input and context
    full_input = f"{input_data}\n\nAdditional Context:\n{context}" if context else input_data

    agent = MetaCognitiveAgent()
    result = await agent.meta_reasoning_loop(full_input)
    
    for key, value in result.items():
        print(f"{key.replace('_', ' ').title()}:")
        print(value)
        print()

if __name__ == "__main__":
    asyncio.run(main())
