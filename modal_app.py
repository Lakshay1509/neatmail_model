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
        "dotenv"
    ])
    
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
        "\""
    )
    .add_local_python_source("main")
)

app = modal.App("email-classifier", image=image)


# ---------------------------------------------------------------------------
# Function config
# ---------------------------------------------------------------------------
@app.function(
    cpu=1,               
    memory=1536,         
    secrets=[modal.Secret.from_name("email-classifier-secrets")],
    timeout=300,
    scaledown_window=60, 
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    from main import app
    return app