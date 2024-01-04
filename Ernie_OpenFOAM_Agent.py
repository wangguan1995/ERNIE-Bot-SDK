import os
import sys 
import asyncio

sys.path.append("./")
os.environ["EB_AGENT_ACCESS_TOKEN"] = "bfa99ff06fbb553c7f9915c5e48ce839502fe7a0"
os.environ["EB_AGENT_LOGGING_LEVEL"] = "info"

from erniebot_agent.tools import RemoteToolkit
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.agents.function_agent_with_retrieval import FunctionAgentWithRetrieval

from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

from sklearn.metrics.pairwise import cosine_similarity

aistudio_access_token = os.environ.get("EB_AGENT_ACCESS_TOKEN", "")
embeddings = ErnieEmbeddings(aistudio_access_token=aistudio_access_token, chunk_size=16)

class FaissSearch:
    def __init__(self, db):
        self.db = db

    def search(self, query: str, top_k: int = 10, **kwargs):
        # 定义一个搜索方法，接受一个查询字符串 'query' 和一个整数 'top_k'，默认为 10
        docs = self.db.similarity_search(query, top_k)
        # 调用数据库的 similarity_search 方法来获取与查询最相关的文档
        para_result = embeddings.embed_documents([i.page_content for i in docs])
        # 对获取的文档内容进行嵌入（embedding），以便进行相似性比较
        query_result = embeddings.embed_query(query)
        # 对查询字符串也进行嵌入
        similarities = cosine_similarity([query_result], para_result).reshape((-1,))
        # 计算查询嵌入和文档嵌入之间的余弦相似度
        retrieval_results = []
        for index, doc in enumerate(docs):
            retrieval_results.append(
                {"content": doc.page_content, "score": similarities[index], "title": doc.metadata["source"]}
            )
        # 遍历每个文档，将内容、相似度得分和来源标题作为字典添加到结果列表中
        return retrieval_results  # 返回包含搜索结果的列表

async def main():
    llm = ERNIEBot(model="ernie-3.5")  # 初始化大语言模型
    tts_tool = RemoteToolkit.from_aistudio("texttospeech").get_tools()  # 获取语音合成工具

    # 创建一个WholeMemory实例。这是一个用于存储对话历史和上下文信息的类，有助于模型理解和持续对话。
    faiss_name = "faiss_index"
    if os.path.exists(faiss_name):
        print(f"\nload {faiss_name} from local")
        db = FAISS.load_local(faiss_name, embeddings)
    else:
        loader = PyPDFDirectoryLoader("construction_regulations")
        documents = loader.load()
        text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=320, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(faiss_name)
    memory = WholeMemory()

    # Vector DB
    faiss_search = FaissSearch(db=db)
    
    # Tool
    tts_tool = RemoteToolkit.from_aistudio("texttospeech").get_tools()[0]

    # Agent
    agent = FunctionAgentWithRetrieval(
        llm=llm, tools=[tts_tool], memory=memory, knowledge_base=faiss_search, threshold=0.9
    )

    # Question 1
    query = "城乡建设部规章中，城市管理执法第三章，第十三条是什么？"
    response = await agent._run(query) # 使用agent的async_run方法来异步执行查询。由于这是异步操作，因此需要使用'await'关键字。
    messages = response.chat_history
    for item in messages:
        print(item.to_dict())

    # Question 2
    query = "把上一轮的检索内容转换为语音"
    response = await agent._run(query)
    messages = response.chat_history
    for item in messages:
        print(item.to_dict())

asyncio.run(main())



## FaissSearch : 搜索 PDF, 按照相关性打分
# from pprint import pprint
# faiss_search = FaissSearch(db=db)
# query = "城市管理执法主管部门的职责是什么？"
# result = faiss_search.search(query=query)
# pprint(result)