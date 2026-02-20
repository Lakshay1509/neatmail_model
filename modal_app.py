import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        extra_options="--index-url https://download.pytorch.org/whl/cpu",
    )
    .pip_install([
        "fastapi[standard]",
        "sentence-transformers",
        "pinecone",
        "openai",
        "uvicorn",
        "python-dotenv",
        "redis"
    ])
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('Qwen/Qwen3-Embedding-0.6B'); "
        "CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
        "\""
    )
    .add_local_python_source("main")
)

app = modal.App("email-classifier", image=image)


@app.cls(
    cpu=2,                      
    memory=2048,                  
    secrets=[modal.Secret.from_name("email-classifier-secrets")],
    timeout=300,
    scaledown_window=300,         
    enable_memory_snapshot=True,
    min_containers=0,             
)
@modal.concurrent(max_inputs=10)
class EmailClassifier:

    @modal.enter(snap=True)
    def load_model(self):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.embedding_model.encode("warmup", normalize_embeddings=True)
        print("Loading cross-encoder reranker...")
        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker_model.predict([("warmup query", "warmup document")])
        print("Models ready.")

    @modal.enter(snap=False)
    def init_clients(self):
        import os
        from pinecone import Pinecone
        from openai import OpenAI
        print("Connecting to Pinecone & OpenAI...")
        self.pinecone_index = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"]
        ).Index(os.environ["PINECONE_INDEX_NAME"])
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        print("Clients ready.")

    @modal.asgi_app()
    def fastapi_app(self):
        from main import app as fastapi_app

        fastapi_app.state.embedding_model = self.embedding_model
        fastapi_app.state.pinecone_index = self.pinecone_index
        fastapi_app.state.openai_client = self.openai_client
        fastapi_app.state.reranker_model = self.reranker_model

        return fastapi_app