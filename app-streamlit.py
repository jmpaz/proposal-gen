import streamlit as st
from streamlit_chat import message

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory


def load_chain():
    llm = OpenAI(model_name="text-davinci-003", temperature=0.8)

    prompt = PromptTemplate(
        input_variables=['history', 'input'],
        output_parser=None,
        template='You are Assistant, a large language model trained by OpenAI and designed to assist with a wide range of tasks, often providing valuable insights and information on a wide range of topics. When needed, messages should be enclosed in an appropriate number of backticks or double quotes, depending on the contents of the input or output message. Please make sure to properly style your responses using Github Flavored Markdown. Use markdown syntax for things like headings, lists, tables, quotes, colored text, code blocks, highlights, superscripts, etc, etc. For emojis, use unicode. Make sure not to mention markdown or styling in your actual response.\n\nCurrent conversation:\n{history}\n\nUser: """""\n{input}"""""\n\nAssistant: ',
        template_format='f-string'
    )

    chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationBufferMemory(human_prefix="User", ai_prefix="Assistant")
    )

    # load initializing prompt from file
    with open("prompt.txt", "r") as f:
        init_prompt = f.read()

    # run the chain with the initializing prompt
    chain.predict(input=init_prompt).strip("<|im_end|>")

    return chain


chain = load_chain()


# Streamlit UI
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "What is your name and function?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input).strip("<|im_end|>")

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")