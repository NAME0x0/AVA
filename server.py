from fastapi import FastAPI
from text_generation import TextGenerator

app = FastAPI()
text_generator = TextGenerator()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/generate_text")
async def generate_text(prompt: str, max_length: int = 100):
    return await text_generator.generate_text(prompt)

@app.on_event("startup")
async def startup_event():
    await text_generator.start_text_generation()

@app.on_event("shutdown")
async def shutdown_event():
    await text_generator.stop_text_generation()
