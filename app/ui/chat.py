"""Chainlit UI for the Lecture Intelligence System.

Implements real-time token streaming and source citations.
All services are called directly in-process (no HTTP dependency).
"""
import os
import uuid
import chainlit as cl

# Chainlit UI hooks below




from app.services.pdf_service import PDFService
from app.services.audio_service import AudioService
from app.services.vector_store import VectorStoreService
from app.services.search_service import SearchService
from app.services.rag_chain import RAGChain
from app.services.llm_engine import stream_answer_from_model


async def process_file(file, collection_name: str):
    msg = cl.Message(content=f"⏳ Processing `{file.name}`... this might take a minute.")
    await msg.send()

    try:
        vector_store = VectorStoreService(collection_name=collection_name)
        extracted = 0
        audio_elements = None

        if file.name.lower().endswith(".pdf"):
            with open(file.path, "rb") as f:
                file_bytes = f.read()

            os.makedirs("data/uploads", exist_ok=True)
            with open(f"data/uploads/{file.name}", "wb") as f:
                f.write(file_bytes)

            import asyncio
            pdf_service = PDFService()
            chunks = await asyncio.to_thread(pdf_service.extract_text_from_pdf, file_bytes, file.name)

            if chunks:
                texts = [c.text for c in chunks]
                metadatas = [{"source": file.name, "page": i + 1} for i in range(len(chunks))]
                ids = [f"{file.name}_chunk_{i}" for i in range(len(chunks))]
                vector_store.add_documents(texts, metadatas, ids)

            extracted = len(chunks)

        elif file.name.lower().endswith((".mp3", ".wav", ".m4a", ".mp4")):
            os.makedirs("data/uploads", exist_ok=True)
            upload_path = f"data/uploads/{file.name}"
            with open(file.path, "rb") as src:
                with open(upload_path, "wb") as dst:
                    dst.write(src.read())

            import asyncio
            audio_service = AudioService()
            segments = await asyncio.to_thread(audio_service.transcribe_audio, upload_path)

            if segments:
                texts = [seg["text"] for seg in segments]
                metadatas = [
                    {"source": file.name, "start": seg["start"], "end": seg["end"]}
                    for seg in segments
                ]
                ids = [f"{file.name}_seg_{i}" for i in range(len(segments))]
                vector_store.add_documents(texts, metadatas, ids)

            extracted = len(segments)
            audio_elements = [
                cl.Audio(name="Lecture Audio", path=upload_path, display="inline")
            ]
        else:
            msg.content = f"❌ Unsupported file type: `{file.name}`"
            await msg.update()
            return False

        msg.content = (
            f"✅ **Successfully processed `{file.name}`!**\n\n"
            f"📄 Extracted **{extracted}** chunks/segments and added them to your knowledge base.\n\n"
            f"You can now ask questions about this material."
        )
        if audio_elements:
            msg.elements = audio_elements

    except Exception as e:
        msg.content = f"❌ Failed to process `{file.name}`. Error: {str(e)}"
        await msg.update()
        return False

    await msg.update()
    return True

@cl.on_chat_start
async def on_chat_start():
    # Create a unique collection for this session so uploads are isolated
    session_id = str(uuid.uuid4())[:8]
    collection_name = f"session_{session_id}"
    cl.user_session.set("collection_name", collection_name)
    cl.user_session.set("thread_metadata", {"collection_name": collection_name})

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="📚 **Welcome to the Lecture Intelligence System!**\n\n"
                    "Upload a PDF or Audio file to begin chatting with your lecture materials.\n"
                    "*(Hint: You can upload more files later using the paperclip icon!)*",
            accept=["application/pdf", "audio/mpeg", "audio/wav",
                    "video/mp4", "audio/mp4", "audio/x-m4a"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]
    await process_file(file, collection_name)

    # Initialize the RAG chain for this session, scoped to this session's collection
    # Reduced top_k to 3 to speed up CPU inference times
    rag = RAGChain(top_k=2, memory_window=5, collection_name=collection_name)
    cl.user_session.set("rag_chain", rag)



@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages with streaming and citations."""
    rag: RAGChain = cl.user_session.get("rag_chain")

    if rag is None:
        await cl.Message(content="⚠️ Please upload a file first before asking questions.").send()
        return

    # Did the user upload new files using the paperclip icon?
    if message.elements:
        collection_name = cl.user_session.get("collection_name")
        for element in message.elements:
            # Check if it has a path (valid file)
            if hasattr(element, "path") and element.path:
                await process_file(element, collection_name)
        
        # We need to refresh the search engine inside rag_chain so it sees the new files
        # Because SearchService loads BM25 into memory on init!
        from app.services.vector_store import VectorStoreService
        from app.services.search_service import SearchService
        vector_store = VectorStoreService(collection_name=collection_name)
        rag.retriever.search_service = SearchService(vector_store)
        
        # If the user only uploaded a file and didn't type text, we can stop here
        if not message.content.strip():
            return

    # Step 1: Retrieve relevant documents
    enriched_question = rag._build_question_with_history(message.content)
    source_docs = rag.retriever.invoke(enriched_question)
    context = rag._format_docs(source_docs)


    # Step 2: Stream the response token by token
    msg = cl.Message(content="")
    await msg.send()

    # Build the chat history string so the LLM can see past conversation turns
    chat_history_str = rag._format_chat_history_for_llm()

    full_answer = ""
    context_list = [context] if context else None
    for token in stream_answer_from_model(query=enriched_question, context=context_list, chat_history=chat_history_str):
        full_answer += token
        await msg.stream_token(token)

    await msg.update()

    # Step 3: Save to chat memory
    rag.chat_history.append({
        "question": message.content,
        "answer": full_answer,
    })

    # Step 4: Display source citations
    if source_docs:
        sources_seen = set()
        citation_lines = []

        for doc in source_docs:
            meta = doc.metadata
            source_name = meta.get("source", "Unknown")

            if "page" in meta:
                cite_key = f"{source_name}_p{meta['page']}"
                if cite_key in sources_seen:
                    continue
                sources_seen.add(cite_key)
                citation_lines.append(f"📄 **{source_name}** — Page {meta['page']}")

            elif "start" in meta:
                start = float(meta["start"])
                cite_key = f"{source_name}_{int(start)}"
                if cite_key in sources_seen:
                    continue
                sources_seen.add(cite_key)
                mins, secs = divmod(int(start), 60)
                timestamp_label = f"{mins}:{secs:02d}"
                # Render a clickable span — custom.js MutationObserver attaches the click handler
                citation_lines.append(
                    f'🎧 **{source_name}** — '
                    f'<span class="seek-timestamp" data-seconds="{start}" '
                    f'style="color:#4fc3f7;cursor:pointer;text-decoration:underline;font-weight:bold;">'
                    f'⏱ Jump to {timestamp_label}</span>'
                )
            else:
                if source_name in sources_seen:
                    continue
                sources_seen.add(source_name)
                citation_lines.append(f"📎 **{source_name}**")

        if citation_lines:
            citation_text = "**📌 Sources:**\n" + "\n".join(citation_lines)
            await cl.Message(content=citation_text).send()


@cl.on_chat_end
async def on_chat_end():
    """Clean up the session's vector store collection when the chat ends."""
    collection_name = cl.user_session.get("collection_name")
    if collection_name:
        try:
            import chromadb
            from app.core.config import settings
            client = chromadb.PersistentClient(path=settings.chroma_dir)
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection may already be gone

