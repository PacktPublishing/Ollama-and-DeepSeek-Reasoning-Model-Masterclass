import streamlit as st
from openai import OpenAI


class LocalCodeAssistant:
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
8. Memory efficiency
Provide specific suggestions for improvements.""",
    }


def get_example_prompts():
    return {
        "Code Generation": {
            "placeholder": """Examples:
1. "Create a Wordle game clone with a simple CLI interface"
2. "Build a PDF merger tool with a progress bar"
3. "Develop a simple REST API for a todo list using FastAPI"
4. "Create a data visualization dashboard using matplotlib"
5. "Build a file encryption tool using Fernet"

Your request:""",
            "default": "Create a tic-tac-toe game with a simple GUI using tkinter",
        },
        "Code Explanation": {
            "placeholder": """Paste your code here for explanation.
Example code to explain:
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
            "default": "",
        },
        "Code Review": {
            "placeholder": """Paste your code here for review.
Example code to review:
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result""",
            "default": "",
        },
    }


def main():
    st.set_page_config(
        page_title="DeepSeek R1 Code Assistant", page_icon="🤖", layout="wide"
    )

    st.title("🤖 Local DeepSeek R1 Code Assistant")
    st.markdown(
        """
    Using DeepSeek R1 1.5B model running locally through Ollama
    """
    )

    system_prompts = get_system_prompts()
    example_prompts = get_example_prompts()

    # Sidebar
    st.sidebar.title("Settings")
    mode = st.sidebar.selectbox(
        "Choose Mode", ["Code Generation", "Code Explanation", "Code Review"]
    )

    # Show current mode description
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Mode**: {mode}")
    st.sidebar.markdown("**Mode Description:**")
    st.sidebar.markdown(system_prompts[mode].replace("\n", "\n\n"))

    # Main content area
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown(f"### Input for {mode}")
        user_prompt = st.text_area(
            "Enter your prompt or code:",
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
                    assistant = LocalCodeAssistant()
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
