# Neural Style Transfer FastAPI Service
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import shutil
from pathlib import Path
import sys
import logging
from typing import Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем путь к текущей директории для импорта style_transfer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Neural Style Transfer Service")

# Настройка шаблонов и статических файлов
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Создаем директории для загрузок и результатов
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Импортируем нашу логику стилизации
from style_transfer import HighQualityStyleTransfer, tensor_to_image

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Главная страница с формой загрузки"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "result_image": None}
    )

@app.post("/transfer")
async def style_transfer(
    request: Request,
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    steps: int = Form(200),
    style_strength: float = Form(1.0)
):
    """Обработка стилизации изображений"""
    try:
        # Генерируем уникальные имена файлов
        content_id = str(uuid.uuid4())
        style_id = str(uuid.uuid4())
        result_id = str(uuid.uuid4())
        
        # Сохраняем загруженные файлы
        content_path = UPLOAD_DIR / f"content_{content_id}{os.path.splitext(content_image.filename)[1]}"
        style_path = UPLOAD_DIR / f"style_{style_id}{os.path.splitext(style_image.filename)[1]}"
        result_path = RESULT_DIR / f"result_{result_id}.jpg"
        
        with open(content_path, "wb") as buffer:
            shutil.copyfileobj(content_image.file, buffer)
        
        with open(style_path, "wb") as buffer:
            shutil.copyfileobj(style_image.file, buffer)
        
        # Выполняем стилизацию
        logger.info(f"Starting style transfer with steps={steps}, style_strength={style_strength}")
        
        transfer = HighQualityStyleTransfer()
        result_tensor = transfer.transfer(
            str(content_path),
            str(style_path),
            steps=steps,
            style_strength=style_strength
        )
        
        # Сохраняем результат
        result_img = tensor_to_image(result_tensor)
        result_img.save(result_path)
        
        logger.info(f"Style transfer completed. Result saved to {result_path}")
        
        # Возвращаем HTML с результатом
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "result_image": f"/results/result_{result_id}.jpg",
                "original_content": f"/uploads/{content_path.name}",
                "original_style": f"/uploads/{style_path.name}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error during style transfer: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": f"Error during style transfer: {str(e)}"}
        )

@app.get("/results/{file_path:path}")
def get_result(file_path: str):
    """Получение результата стилизации"""
    return FileResponse(RESULT_DIR / file_path)

@app.get("/uploads/{file_path:path}")
def get_upload(file_path: str):
    """Получение загруженных изображений"""
    return FileResponse(UPLOAD_DIR / file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)