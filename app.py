from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import os


app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
@app.post("/train")
async def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"

@app.post("/predict")
async def predictRoute(request: Request):
    data = await request.json()
    image = data['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return JSONResponse(content=result)

if __name__ == "__main__":
    clApp = ClientApp()
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080) #for AWS
