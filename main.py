# Neural Style Transfer FastAPI Service
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import shutil
from pathlib import Path
import sys
import logging
import threading
from typing import Optional, Dict
import json
from datetime import datetime
import mlflow
from prometheus_client import generate_latest, Counter, Histogram, Enum # Импорт Prometheus
from starlette.responses import Response # Импорт Response для эндпоинта /metrics

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
TASKS_DIR = Path("tasks")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
TASKS_DIR.mkdir(exist_ok=True)

# Настройка MLflow (выполняется при запуске через uvicorn)
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("Neural Style Transfer")

# Проверяем, что директории доступны для записи
try:
    test_file = UPLOAD_DIR / ".test_write"
    test_file.write_text("test")
    test_file.unlink()
    
    test_file = RESULT_DIR / ".test_write"
    test_file.write_text("test")
    test_file.unlink()
    
    test_file = TASKS_DIR / ".test_write"
    test_file.write_text("test")
    test_file.unlink()
    
    logger.info("All directories are writable")
except Exception as e:
    logger.error(f"Directory permission error: {str(e)}")
    raise

# Импортируем нашу логику стилизации
from style_transfer import HighQualityStyleTransfer, tensor_to_image

# Глобальный словарь для отслеживания задач
tasks_status: Dict[str, Dict] = {}
# Блокировка для потокобезопасного доступа к tasks_status
tasks_lock = threading.Lock()

# Метрики Prometheus
TASK_STATUS_METRIC = Enum(
    'style_transfer_task_status', 'Status of style transfer tasks',
    ['task_id'],
    states=['queued', 'processing', 'completed', 'failed']
)
TASK_DURATION_SECONDS = Histogram(
    'style_transfer_task_duration_seconds', 'Duration of style transfer tasks in seconds',
    ['task_id', 'status']
)

# Эндпоинт для метрик Prometheus
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


# Функция для выполнения стилизации в фоне
def run_style_transfer(task_id: str, content_path: str, style_path: str, result_path: str, steps: int, style_strength: float):
    """Выполняет стилизацию в фоновом режиме и обновляет статус"""
    logger.info(f"RUN_STYLE_TRANSFER CALLED with task_id={task_id}")
    logger.info(f"Content path exists: {os.path.exists(content_path)}")
    logger.info(f"Style path exists: {os.path.exists(style_path)}")
    logger.info(f"Result dir exists: {os.path.exists(RESULT_DIR)}")
    logger.info(f"Result dir writable: {os.access(RESULT_DIR, os.W_OK)}")
    
    logger.info(f"Starting style transfer task {task_id} with params: content_path={content_path}, style_path={style_path}, result_path={result_path}, steps={steps}, style_strength={style_strength}")
    
    start_task_time = datetime.now() # Замеряем время начала выполнения задачи

    try:
        # Начинаем MLflow run
        with mlflow.start_run(run_name=f"style_transfer_{task_id}"):
            try:
                TASK_STATUS_METRIC.labels(task_id=task_id).state('processing')
                # Обновляем статус задачи с использованием блокировки
                with tasks_lock:
                    tasks_status[task_id] = {
                        "status": "processing",
                        "progress": 0,
                        "start_time": start_task_time.isoformat(),
                        "content_path": content_path,
                        "style_path": style_path,
                        "result_path": None,
                        "error": None
                    }

                # Сохраняем статус в файл
                with open(TASKS_DIR / f"{task_id}.json", "w") as f:
                    json.dump(tasks_status[task_id], f)

                logger.info(f"Initializing HighQualityStyleTransfer for task {task_id}")
                transfer = HighQualityStyleTransfer()

                # Функция для обновления прогресса
                def update_progress(current_step, total_steps):
                    progress = int((current_step / total_steps) * 100)
                    with tasks_lock:
                        tasks_status[task_id]["progress"] = progress
                        tasks_status[task_id]["status"] = "processing"
                    logger.debug(f"Task {task_id} progress: {progress}%")

                    # Логируем прогресс в MLflow
                    mlflow.log_metric("progress", progress, step=current_step)

                    # Сохраняем статус в файл
                    with open(TASKS_DIR / f"{task_id}.json", "w") as f:
                        json.dump(tasks_status[task_id], f)

                logger.info(f"Starting transfer for task {task_id}")
                # Выполняем стилизацию с callback для обновления прогресса
                result_tensor = transfer.transfer(
                    content_path,
                    style_path,
                    steps=steps,
                    style_strength=style_strength,
                    callback=update_progress,
                    task_id=task_id
                )
                logger.info(f"Transfer completed for task {task_id}")

                # Сохраняем результат
                try:
                    result_img = tensor_to_image(result_tensor)
                    result_img.save(result_path)

                    # Проверяем, что файл действительно сохранился
                    if not os.path.exists(result_path):
                        logger.error(f"Result file was not saved: {result_path}")
                        raise Exception(f"Failed to save result file: {result_path}")

                    logger.info(f"Result successfully saved to {result_path}")

                    # Обновляем статус с использованием блокировки
                    result_filename = os.path.basename(result_path)
                    with tasks_lock:
                        tasks_status[task_id].update({
                            "status": "completed",
                            "progress": 100,
                            "end_time": datetime.now().isoformat(),
                            "result_path": str(result_path),
                            "result_filename": result_filename
                        })

                    # Сохраняем статус в файл
                    with open(TASKS_DIR / f"{task_id}.json", "w") as f:
                        json.dump(tasks_status[task_id], f, indent=2)

                    logger.info(f"Task {task_id} status updated to completed")
                    TASK_STATUS_METRIC.labels(task_id=task_id).state('completed')
                    TASK_DURATION_SECONDS.labels(task_id=task_id, status='completed').observe((datetime.now() - start_task_time).total_seconds())

                except Exception as e:
                    logger.error(f"Error saving result: {str(e)}", exc_info=True)
                    with tasks_lock:
                        tasks_status[task_id].update({
                            "status": "failed",
                            "progress": 0,
                            "error": f"Error saving result: {str(e)}",
                            "end_time": datetime.now().isoformat()
                        })

                    # Сохраняем статус в файл
                    with open(TASKS_DIR / f"{task_id}.json", "w") as f:
                        json.dump(tasks_status[task_id], f, indent=2)

                    # Логируем метрики в MLflow
                    mlflow.log_metric("task_duration_seconds", (datetime.now() - start_task_time).total_seconds())
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error_message", str(e))

                    TASK_STATUS_METRIC.labels(task_id=task_id).state('failed')
                    TASK_DURATION_SECONDS.labels(task_id=task_id, status='failed').observe((datetime.now() - start_task_time).total_seconds())

            except Exception as e:
                logger.error(f"Error during style transfer for task {task_id}: {str(e)}", exc_info=True)
                with tasks_lock:
                    tasks_status[task_id] = {
                        "status": "failed",
                        "progress": 0,
                        "error": str(e),
                        "content_path": content_path,
                        "style_path": style_path,
                        "result_path": None,
                        "end_time": datetime.now().isoformat()
                    }

                # Логируем ошибку в MLflow
                mlflow.log_metric("task_duration_seconds", (datetime.now() - start_task_time).total_seconds())
                mlflow.set_tag("status", "failed")
                if "error" in tasks_status[task_id]:
                    mlflow.log_param("error_message", tasks_status[task_id]["error"])

                # Сохраняем статус в файл
                try:
                    with open(TASKS_DIR / f"{task_id}.json", "w") as f:
                        json.dump(tasks_status[task_id], f, indent=2)
                    logger.info(f"Task {task_id} status updated to failed")
                except Exception as file_error:
                    logger.error(f"Error saving task status: {str(file_error)}", exc_info=True)
                TASK_DURATION_SECONDS.labels(task_id=task_id, status='failed').observe((datetime.now() - start_task_time).total_seconds())
                TASK_STATUS_METRIC.labels(task_id=task_id).state('failed')

    except Exception as outer_e:
        # Если сломается сам MLflow (например, нет прав на папку mlruns)
        logger.error(f"Critical MLflow error in task {task_id}: {outer_e}", exc_info=True)
        with tasks_lock:
            tasks_status[task_id] = {
                "status": "failed",
                "progress": 0,
                "error": f"Task init error: {str(outer_e)}",
                "end_time": datetime.now().isoformat()
            }
        with open(TASKS_DIR / f"{task_id}.json", "w") as f:
            json.dump(tasks_status[task_id], f)


@app.post("/transfer")
async def style_transfer(
    request: Request,
    background_tasks: BackgroundTasks,
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    steps: int = Form(200),
    style_strength: float = Form(1.0)
):
    """Запускает стилизацию изображений в фоновом режиме"""
    try:
        # Генерируем уникальные идентификаторы
        task_id = str(uuid.uuid4())
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
        
        # Инициализируем статус задачи
        tasks_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "start_time": datetime.now().isoformat(),
            "content_path": str(content_path),
            "style_path": str(style_path),
            "result_path": None,
            "error": None
        }
        
        # Сохраняем статус в файл
        with open(TASKS_DIR / f"{task_id}.json", "w") as f:
            json.dump(tasks_status[task_id], f)
        
        TASK_STATUS_METRIC.labels(task_id=task_id).state('queued')
        # Запускаем обработку в отдельном потоке
        logger.info(f"Starting thread for task {task_id} with params: content_path={content_path}, style_path={style_path}, result_path={result_path}, steps={steps}, style_strength={style_strength}")
        
        try:
            thread = threading.Thread(
                target=run_style_transfer,
                args=(task_id, str(content_path), str(style_path), str(result_path), steps, style_strength),
                name=f"StyleTransfer-{task_id}",
                daemon=True
            )
            thread.start()
            logger.info(f"Task {task_id} started in thread {thread.name}")
        except Exception as e:
            logger.error(f"Error starting thread for task {task_id}: {str(e)}", exc_info=True)
            with tasks_lock:
                tasks_status[task_id].update({
                    "status": "failed",
                    "error": f"Error starting style transfer: {str(e)}"
                })
            with open(TASKS_DIR / f"{task_id}.json", "w") as f:
                json.dump(tasks_status[task_id], f)
            return JSONResponse(
                status_code=500,
                content={"message": f"Error starting style transfer: {str(e)}"}
            )
        
        # Перенаправляем на страницу статуса
        return RedirectResponse(url=f"/status/{task_id}", status_code=303)
        
    except Exception as e:
        logger.error(f"Error starting style transfer: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": f"Error starting style transfer: {str(e)}"}
        )

@app.get("/results/{file_path:path}")
def get_result(file_path: str):
    """Получение результата стилизации"""
    logger.info(f"Request for result file: {file_path}")
    logger.info(f"RESULT_DIR: {RESULT_DIR}")
    
    result_path = RESULT_DIR / file_path
    logger.info(f"Full result path: {result_path}")
    logger.info(f"File exists: {result_path.exists()}")
    
    if not result_path.exists():
        logger.error(f"Result file not found: {result_path}")
        
        # Если файл не найден, ищем задачу с таким результатом
        task_files = list(TASKS_DIR.glob("*.json"))
        logger.info(f"Found {len(task_files)} task files")
        
        for task_file in task_files:
            try:
                with open(task_file, "r") as f:
                    task_status = json.load(f)
                    logger.info(f"Checking task {task_file.stem}: {task_status.get('result_path')}")
                    if task_status.get("result_path") and os.path.basename(task_status["result_path"]) == file_path:
                        logger.warning(f"Task found but result file missing: {task_file}")
                        # Если задача найдена, но файл результата отсутствует, возвращаем ошибку
                        raise HTTPException(
                            status_code=404, 
                            detail=f"Result file not found: {file_path}"
                        )
            except Exception as e:
                logger.error(f"Error reading task file {task_file}: {str(e)}")
        
        # Если задача не найдена
        logger.error(f"No task found for result: {file_path}")
        raise HTTPException(status_code=404, detail="Result not found")
    
    logger.info(f"Serving result file: {result_path}")
    return FileResponse(result_path)

@app.get("/status/{task_id}")
async def get_status(request: Request, task_id: str):
    """Страница статуса задачи"""
    # Проверяем статус задачи
    task_file = TASKS_DIR / f"{task_id}.json"
    
    if not task_file.exists():
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "Task not found"},
            status_code=404
        )
    
    # Читаем статус задачи
    with open(task_file, "r") as f:
        task_status = json.load(f)
    
    # Если задача завершена, перенаправляем на результат
    if task_status.get("status") == "completed":
        result_path = task_status.get("result_path")
        if result_path:
            # Используем только имя файла результата
            result_filename = os.path.basename(result_path)
            logger.info(f"Redirecting to result: {result_filename}")
            return RedirectResponse(url=f"/?result={result_filename}", status_code=303)
    
    # Если задача завершилась с ошибкой
    if task_status.get("status") == "failed":
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": task_status.get("error", "Unknown error")},
            status_code=500
        )
    
    # Возвращаем страницу статуса
    return templates.TemplateResponse(
        "status.html",
        {
            "request": request,
            "task_id": task_id,
            "status": task_status.get("status", "unknown"),
            "progress": task_status.get("progress", 0),
            "content_image": f"/uploads/{os.path.basename(task_status['content_path'])}",
            "style_image": f"/uploads/{os.path.basename(task_status['style_path'])}"
        }
    )

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """API эндпоинт для получения статуса задачи"""
    task_file = TASKS_DIR / f"{task_id}.json"
    
    if not task_file.exists():
        logger.warning(f"Task file not found: {task_file}")
        return JSONResponse(
            status_code=404,
            content={"error": "Task not found"}
        )
    
    try:
        with open(task_file, "r") as f:
            task_status = json.load(f)
            
        # Добавляем дополнительные поля для отладки
        response_data = {
            "status": task_status.get("status", "unknown"),
            "progress": task_status.get("progress", 0),
            "result_path": task_status.get("result_path"),
            "result_filename": task_status.get("result_filename"),
            "error": task_status.get("error"),
            "task_id": task_id
        }
        
        logger.debug(f"Task status response: {response_data}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error reading task status: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error reading task status: {str(e)}"}
        )

@app.get("/uploads/{file_path:path}")
def get_upload(file_path: str):
    """Получение загруженных изображений"""
    return FileResponse(UPLOAD_DIR / file_path)

@app.get("/")
async def read_root(request: Request, result: Optional[str] = None):
    """Главная страница с формой загрузки"""
    if result:
        logger.info(f"Processing result parameter: {result}")
        
        # Проверяем, существует ли файл результата
        result_path = RESULT_DIR / result
        logger.info(f"Checking result file: {result_path}")
        
        if not result_path.exists():
            logger.error(f"Result file not found: {result_path}")
            
            # Если файл не найден, ищем задачу с таким результатом
            task_files = list(TASKS_DIR.glob("*.json"))
            for task_file in task_files:
                try:
                    with open(task_file, "r") as f:
                        task_status = json.load(f)
                        if task_status.get("result_path") and os.path.basename(task_status["result_path"]) == result:
                            logger.info(f"Found task for result: {task_file}")
                            # Если задача найдена, но файл отсутствует, возвращаемся на страницу статуса
                            return RedirectResponse(url=f"/status/{task_file.stem}", status_code=303)
                except Exception as e:
                    logger.error(f"Error reading task file {task_file}: {str(e)}")
            
            # Если задача не найдена, показываем ошибку
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "message": f"Result file not found: {result}"},
                status_code=404
            )
        
        # Получаем оригинальные изображения для отображения
        original_content = None
        original_style = None
        
        task_files = list(TASKS_DIR.glob("*.json"))
        for task_file in task_files:
            try:
                with open(task_file, "r") as f:
                    task_status = json.load(f)
                    if task_status.get("result_path") and os.path.basename(task_status["result_path"]) == result:
                        original_content = f"/uploads/{os.path.basename(task_status['content_path'])}"
                        original_style = f"/uploads/{os.path.basename(task_status['style_path'])}"
                        logger.info(f"Found original images: content={original_content}, style={original_style}")
                        break
            except Exception as e:
                logger.error(f"Error reading task file {task_file}: {str(e)}")
        
        # Проверяем, что result_image - это правильный URL
        result_image = f"/results/{result}"
        logger.info(f"Displaying result: {result_image}")
        
        # Добавляем timestamp для предотвращения кеширования
        import time
        timestamp = int(time.time())
        
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "result_image": result_image,
                "original_content": original_content,
                "original_style": original_style,
                "timestamp": timestamp
            }
        )
    
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "result_image": None}
    )

if __name__ == "__main__":
    # # Настройка MLflow
    # mlflow.set_experiment("Neural Style Transfer")
    # mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    #
    # # Создаем директорию для MLflow, если она не существует
    # os.makedirs("mlruns", exist_ok=True)
    
    import uvicorn
    # Используем более простую конфигурацию
    uvicorn.run(app, host="0.0.0.0", port=8000)