import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

# Carrega o arquivo CSV com dados dos personagens
loader = CSVLoader(file_path="characters.csv")
characters = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(characters, embeddings)


def retrieve_characteristics(characteristics):
    similar = db.similarity_search(characteristics, k=3)
    return [doc.page_content for doc in similar]


# Chat model configuration
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
Você é tira dúvidas de fãs de StarWars
Sua função é de responder com um ou mais personagens que atendem a descrição
Vou lhe passar algumas descrições e seus personagens específicos

Siga as regras abaixo:
1/ Vc deve buscar encontrar e responder com base nos personagens da base de dados
2/ Suas respostas devem ser de no mínimo 1 personagem

Aqui está a descrição fornecida pelo cliente
{message}

Aqui está uma lista de carcaterísitcas e personagens para ter como base
{best_practice}

Escreva a melhor resposta que eu deveria enviar para esse cliente:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


def generate_character_response(characteristics):
    similar_characters = retrieve_characteristics(characteristics)
    return similar_characters


def main():
    st.set_page_config(
        page_title="Character Guesser", page_icon=":mage:")
    st.header("Character Guesser")

    characteristics = st.text_area("Enter character features")

    if characteristics:
        st.write("Searching for characters matching the features...")

        results = generate_character_response(characteristics)

        st.info(", ".join(results))


if __name__ == '__main__':
    main()
