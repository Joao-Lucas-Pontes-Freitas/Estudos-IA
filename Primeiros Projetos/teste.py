import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

load_dotenv()

loader = CSVLoader(file_path="FAQ_Starlink.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar]


retrieve_info("Qual e o custo do servico Starlink?")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
Você é assistente de vendas da Starlink
Sua função é de respopnder as dúvidas que recebemos
Vou lhe passar algumas perguntas e respostas para vc ter como base

Siga as regras abaixo:
1/ Vc deve buscar se comportar com oas respostas que foram forncedidas de base, 
2/ Suas respostas devem ser similares em tom, comrpimento e argumento da base fornecida
3/ Alguns dos email podem conter links e informações irrelevantes. Preste atenção somente no conteúdo da mensagem

Aqui esta a mensagem recebida do cliente.
{message}

Aqui está uma lista de perguntas a respostas de histórico que serve de base:
{best_practice}

Escreva a melhor resposta que eu deveria enviar para esse cliente:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


def main():
    st.set_page_config(
        page_title="E-mail manager", page_icon=":bird:")
    st.header("Email manager")
    message = st.text_area("Email do cliente")

    if message:
        st.write("Gerando o email de respsota baseado nas melhores práticas...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
