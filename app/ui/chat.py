import chainlit as cl
import requests
import os

API_URL = "http://localhost:8000/api"

@cl.on_chat_start
async def on_chat_start():
    files = None
    
    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Welcome to the Lecture Intelligence System! Please upload a PDF or Audio file to begin.",
            accept=["application/pdf", "audio/mpeg", "audio/wav", "video/mp4", "audio/mp4", "audio/x-m4a"],
            max_size_mb=50,
            timeout=180,
        ).send()

    file = files[0]
    
    msg = cl.Message(content=f"Processing `{file.name}`... this might take a minute.")
    await msg.send()
    
    # Send file to FastAPI
    with open(file.path, "rb") as f:
        files_data = {"file": (file.name, f, "multipart/form-data")}
        try:
            # We must use proper multipart requests so we pass 'files'
            # FastApi expects a form data 'file'
            import httpx
            with httpx.Client() as client:
                response = client.post(f"{API_URL}/upload", files={"file": (file.name, f)})
            response.raise_for_status()
            data = response.json()
            
            extracted = data.get("chunks_extracted") or data.get("segments_extracted", 0)
            msg.content = f"Successfully processed `{file.name}`! Extracted {extracted} chunks/segments."
        except Exception as e:
            msg.content = f"Failed to process `{file.name}`. Error: {str(e)}"
    
    await msg.update()

@cl.on_message
async def on_message(message: cl.Message):
    # Pass query to the API
    try:
        import httpx
        with httpx.Client() as client:
            response = client.post(
                f"{API_URL}/chat",
                json={"query": message.content, "top_k": 5}
            )
        response.raise_for_status()
        answer = response.json().get("answer", "No answer received.")
        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"Error contacting API: {str(e)}").send()
