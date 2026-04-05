# server/app.py — OpenEnv multi-mode entry point
import uvicorn
from customer_support_env.server import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()