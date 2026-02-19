"""Gradio app and CLI entry point for the RAG chatbot."""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_gradio():
    """Launch Gradio UI."""
    import gradio as gr
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from src.ingestion import get_or_create_vector_store
    from src.rag_chain import query_rag
    from src.memory import extract_and_write_memory

    def index_files(files):
        """Index uploaded .txt files."""
        if not files:
            return "No files selected."
        file_list = files if isinstance(files, list) else [files]
        docs_dir = ROOT / "sample_docs"
        docs_dir.mkdir(exist_ok=True)
        count = 0
        for f in file_list:
            if not f:
                continue
            src_path = f if isinstance(f, (str, Path)) else getattr(f, "name", f)
            src_path = Path(src_path)
            if src_path.suffix.lower() == ".txt":
                dest = docs_dir / src_path.name
                with open(src_path, "rb") as src:
                    dest.write_bytes(src.read())
                count += 1
        if count == 0:
            return "No .txt files to index."
        try:
            get_or_create_vector_store(docs_dir, force_recreate=True)
            return f"Indexed {count} file(s)."
        except Exception as e:
            return f"Error: {e}"

    def index_sample():
        """Index sample_docs directory."""
        try:
            get_or_create_vector_store(force_recreate=True)
            return "Indexed sample_docs."
        except Exception as e:
            return f"Error: {e}"

    def chat(message, history):
        """Handle chat message and return response with citations."""
        history = history or []
        if not message or not message.strip():
            return history

        try:
            answer, citations = query_rag(message.strip(), k=5)
        except Exception as e:
            answer = f"Error: {e}"
            citations = []

        # Format response with citations
        if citations:
            cites_text = "\n\n**Citations:**\n"
            for i, c in enumerate(citations, 1):
                snippet = c["snippet"][:200] + "..." if len(c["snippet"]) > 200 else c["snippet"]
                cites_text += f"\n{i}. **{c['source']}** ({c['locator']}): {snippet}"
            full_response = answer + cites_text
        else:
            full_response = answer

        # Memory extraction
        if answer and not answer.startswith("Error:"):
            try:
                extract_and_write_memory(message, answer)
            except Exception:
                pass

        # Gradio 6 expects messages with role and content
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_response},
        ]

    with gr.Blocks(title="Agentic RAG Chatbot") as demo:
        gr.Markdown("# Agentic RAG Chatbot")
        gr.Markdown("File-grounded Q&A with citations and persistent memory.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Documents")
                file_upload = gr.File(
                    label="Upload .txt files",
                    file_types=[".txt"],
                    file_count="multiple",
                )
                index_btn = gr.Button("Index uploaded files")
                index_sample_btn = gr.Button("Index sample_docs")
                index_status = gr.Textbox(label="Status", interactive=False)

                index_btn.click(
                    fn=index_files,
                    inputs=[file_upload],
                    outputs=[index_status],
                )
                index_sample_btn.click(
                    fn=index_sample,
                    outputs=[index_status],
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                )
                msg = gr.Textbox(
                    label="Ask a question",
                    placeholder="Ask about your documents...",
                    show_label=False,
                    container=False,
                )
                submit = gr.Button("Submit")

        def respond(message, history):
            return chat(message, history)

        msg.submit(respond, [msg, chatbot], [chatbot]).then(
            lambda: "", None, [msg]
        )
        submit.click(respond, [msg, chatbot], [chatbot]).then(
            lambda: "", None, [msg]
        )

    demo.launch(server_name="127.0.0.1", theme=gr.themes.Soft())


def run_cli():
    """CLI mode for make sanity and scripting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index sample_docs and exit")
    parser.add_argument("--question", "-q", type=str, help="Ask a question")
    parser.add_argument("--cli", action="store_true", help="Run one question and print answer")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    if args.index:
        from src.ingestion import get_or_create_vector_store
        get_or_create_vector_store(force_recreate=True)
        print("Indexed.")
        return

    if args.question or args.cli:
        from src.rag_chain import query_rag
        q = args.question or "Summarize the main contribution in 3 bullets."
        answer, citations = query_rag(q)
        print("Answer:", answer)
        if citations:
            print("\nCitations:")
            for c in citations:
                print(f"  - {c['source']} ({c['locator']}): {c['snippet'][:100]}...")
        return

    # Default: run Gradio
    run_gradio()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--index", "--question", "-q", "--cli"):
        run_cli()
    else:
        run_gradio()
