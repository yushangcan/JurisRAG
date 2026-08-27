"""Start the online API from the project root: python run_api.py"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("online_retrieval.api:app", host="127.0.0.1", port=8000, reload=False)
