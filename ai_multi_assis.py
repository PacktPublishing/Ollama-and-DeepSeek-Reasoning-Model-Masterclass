import streamlit as st
from openai import OpenAI


class MultiAssistant:
    def __init__(self):
        self.client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1/",
        )
        self.model = "deepseek-r1:1.5b"

    def process_request(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )

            result = st.empty()
            collected_chunks = []

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    collected_chunks.append(chunk.choices[0].delta.content)
                    result.markdown("".join(collected_chunks))

            return "".join(collected_chunks)

        except Exception as e:
            return f"Error: {str(e)}"


def get_system_prompts():
    return {
        # Code Assistant Prompts
        "Code Generation": """You are an expert Python programmer who writes clean, efficient, and well-documented code.
Follow these guidelines:
1. Start with a brief comment explaining the code's purpose
2. Include docstrings for functions
3. Use clear variable names
4. Add inline comments for complex logic
5. Follow PEP 8 style guidelines
6. Include example usage
7. Handle common edge cases""",
        "Code Explanation": """You are a patient and knowledgeable coding tutor.
Analyze the code and explain:
1. Overall purpose and functionality
2. Break down of each major component
3. Key programming concepts used
4. Flow of execution
5. Important variables and functions
6. Any clever techniques or patterns
7. Potential learning points for students""",
        "Code Review": """You are a senior code reviewer with expertise in Python best practices.
Review the code for:
1. Logical errors or bugs
2. Performance optimization opportunities
3. Security vulnerabilities
4. Code style and PEP 8 compliance
5. Error handling improvements
6. Documentation completeness
7. Code modularity and reusability
8. Memory efficiency""",
        # Language Tutor Prompts
        "Grammar Check": """You are an expert English language teacher.
Review the text for:
1. Grammar errors
2. Punctuation mistakes
3. Sentence structure
4. Word choice improvements
5. Style consistency
Provide clear explanations and corrections.""",
        "Vocabulary Enhancement": """You are a vocabulary expert.
Analyze the text and:
1. Suggest more sophisticated alternatives
2. Explain idioms and phrases
3. Provide context for word usage
4. Suggest synonyms and antonyms
5. Explain connotations""",
        # Document Generator Prompts
        "Business Proposal": """You are a professional business writer.
Generate a proposal that includes:
1. Executive summary
2. Problem statement
3. Proposed solution
4. Timeline and milestones
5. Budget breakdown
6. Risk assessment
7. Expected outcomes""",
        "Professional Email": """You are an expert in business communication.
Create an email that:
1. Has a clear subject line
2. Maintains professional tone
3. Is concise and focused
4. Includes call to action
5. Has appropriate closing
6. Follows email etiquette""",
    }


def get_example_prompts():
    return {
        # Code Assistant Examples
        "Code Generation": {
            "placeholder": "Create a tic-tac-toe game with GUI using tkinter",
            "default": "",
        },
        "Code Explanation": {
            "placeholder": "Paste code here for explanation",
            "default": "",
        },
        "Code Review": {"placeholder": "Paste code here for review", "default": ""},
        # Language Tutor Examples
        "Grammar Check": {"placeholder": "Paste text for grammar check", "default": ""},
        "Vocabulary Enhancement": {
            "placeholder": "Paste text for vocabulary improvement",
            "default": "",
        },
        # Document Generator Examples
        "Business Proposal": {
            "placeholder": "Describe the business proposal requirements",
            "default": "",
        },
        "Professional Email": {
            "placeholder": "Describe the email context and requirements",
            "default": "",
        },
    }


def main():
    st.set_page_config(page_title="AI Multi-Assistant", page_icon="🤖", layout="wide")

    st.title("🤖 AI Multi-Assistant")

    # Sidebar for tool selection
    st.sidebar.title("Tool Selection")

    tool_categories = {
        "Code Assistant": ["Code Generation", "Code Explanation", "Code Review"],
        "Language Tutor": ["Grammar Check", "Vocabulary Enhancement"],
        "Document Generator": ["Business Proposal", "Professional Email"],
    }

    category = st.sidebar.selectbox("Select Category", list(tool_categories.keys()))
    mode = st.sidebar.selectbox("Select Tool", tool_categories[category])

    system_prompts = get_system_prompts()
    example_prompts = get_example_prompts()

    # Show current mode description
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Tool**: {mode}")
    st.sidebar.markdown("**Tool Description:**")
    st.sidebar.markdown(system_prompts[mode].replace("\n", "\n\n"))

    # Main content area
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown(f"### Input for {mode}")
        user_prompt = st.text_area(
            "Enter your prompt:",
            height=300,
            placeholder=example_prompts[mode]["placeholder"],
            value=example_prompts[mode]["default"],
        )

        process_button = st.button(
            "🚀 Process", type="primary", use_container_width=True
        )

    with col2:
        st.markdown("### Output")
        output_container = st.container()

    if process_button:
        if user_prompt:
            with st.spinner("Processing..."):
                with output_container:
                    assistant = MultiAssistant()
                    assistant.process_request(system_prompts[mode], user_prompt)
        else:
            st.warning("⚠️ Please enter some input!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center'>
        <p>Made with ❤️ using DeepSeek R1 and Ollama</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
