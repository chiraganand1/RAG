import argparse
from rag.index import build_index
from rag.generate import answer
from dotenv import load_dotenv

load_dotenv()   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path to the PDF to index")
    parser.add_argument("--build", action="store_true", help="Build the index from --pdf")
    parser.add_argument("--ask", help="Ask a question against the already-built index")
    parser.add_argument("--top_k", type=int, default=5, help="How many chunks to retrieve")
    args = parser.parse_args()

    if args.build:
        if not args.pdf:
            raise SystemExit("--build requires --pdf <path>")
        build_index(args.pdf)
    elif args.ask:
        print(answer(args.ask, top_k=args.top_k))
    else:
        print("Use --build --pdf <path> first, then --ask '<question>'")