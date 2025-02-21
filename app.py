from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
from enum import Enum
import os
import sys
import asyncio
import hashlib
import logging
from datetime import datetime
import psutil
from xml.etree import ElementTree
import requests
from openai import AsyncOpenAI
from aiofiles import open as aio_open
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

app = FastAPI(title="Document Crawler and Processor API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
COMBINED_DIR = os.path.join(BASE_DIR, "combined")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

for directory in [OUTPUT_DIR, COMBINED_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

class CrawlRequest(BaseModel):
    sitemap_url: HttpUrl
    max_concurrent: int = 10
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    CRAWLING = "crawling"
    CRAWL_COMPLETE = "crawl_complete"
    AI_PROCESSING = "ai_processing"
    AI_COMPLETE = "ai_complete"
    COMBINING = "combining"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStatus(BaseModel):
    job_id: str
    status: ProcessingStatus
    message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_urls: Optional[int] = None
    crawled_urls: int = 0
    ai_processed_urls: int = 0
    failed_urls: List[str] = []

job_statuses: Dict[str, JobStatus] = {}

def get_safe_filename(url: str) -> str:
    return f"{hashlib.md5(url.encode('utf-8')).hexdigest()}.md"

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]
        return urls
    except Exception as e:
        logger.error(f"Error fetching sitemap: {e}")
        return []

async def process_with_openai(content: str, api_key: str, model: str) -> str:
    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": """Process this documentation for a RAG system by:
                    1. Removing any marketing language or fluff
                    2. Preserving all technical information and code examples
                    3. Maintaining the document structure but removing redundant sections
                    5. Making the content clear and concise
                    6. Keeping all essential technical details and examples
                    CRITICAL: DO NOT ADD ANY EXTRA DATA TO THE DOCUMENT OR COMMENT ABOUT IT"""
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI processing error: {e}")
        return content

async def crawl_parallel(urls: List[str], max_concurrent: int, job_id: str):
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator()
    )
    
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        success_count = 0
        failed_urls = []
        
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i:i + max_concurrent]
            tasks = []
            
            for j, url in enumerate(batch):
                session_id = f"session_{job_id}_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Error crawling {url}: {result}")
                    failed_urls.append(url)
                elif result.success:
                    success_count += 1
                    markdown_output = result.markdown_v2.raw_markdown
                    filename = get_safe_filename(url)
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    async with aio_open(file_path, "w", encoding="utf-8") as f:
                        await f.write(markdown_output)
                    job_statuses[job_id].crawled_urls += 1
                else:
                    failed_urls.append(url)
                    logger.error(f"Failed crawling {url}: {result.error_message}")

        return success_count, failed_urls

    finally:
        await crawler.close()

async def process_documents_parallel(job_id: str, openai_api_key: str, openai_model: str, max_concurrent: int = 10):
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.md')]
    processed_files = []
    
    async def process_file(filename):
        input_path = os.path.join(OUTPUT_DIR, filename)
        output_path = os.path.join(PROCESSED_DIR, filename)
        
        try:
            async with aio_open(input_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            processed_content = await process_with_openai(content, openai_api_key, openai_model)
            
            async with aio_open(output_path, 'w', encoding='utf-8') as f:
                await f.write(processed_content)
            
            job_statuses[job_id].ai_processed_urls += 1
            return output_path
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            async with aio_open(input_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            async with aio_open(output_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            job_statuses[job_id].ai_processed_urls += 1
            return output_path

    tasks = []
    for i in range(0, len(files), max_concurrent):
        batch = files[i:i + max_concurrent]
        batch_tasks = [process_file(filename) for filename in batch]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        for result in results:
            if not isinstance(result, Exception):
                processed_files.append(result)
    
    return processed_files

async def combine_documents(job_id: str) -> str:
    files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('.md')])
    if not files:
        raise Exception("No processed files to combine")

    combined_content = []
    for filename in files:
        file_path = os.path.join(PROCESSED_DIR, filename)
        async with aio_open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            combined_content.append(content)

    output_file = f"combined_{job_id}.md"
    output_path = os.path.join(COMBINED_DIR, output_file)
    
    async with aio_open(output_path, 'w', encoding='utf-8') as f:
        await f.write('\n\n---\n\n'.join(combined_content))
    
    return output_path

async def process_crawl_request(job_id: str, request: CrawlRequest):
    try:
        # Crawling phase
        job_statuses[job_id].status = ProcessingStatus.CRAWLING
        urls = get_urls_from_sitemap(str(request.sitemap_url))
        
        if not urls:
            raise Exception("No URLs found in sitemap")
            
        job_statuses[job_id].total_urls = len(urls)
        success_count, failed_urls = await crawl_parallel(urls, request.max_concurrent, job_id)
        job_statuses[job_id].failed_urls = failed_urls

        if success_count == 0:
            raise Exception("No documents were successfully crawled")

        job_statuses[job_id].status = ProcessingStatus.CRAWL_COMPLETE
        job_statuses[job_id].message = f"Crawling completed. {success_count} URLs crawled."

        # OpenAI processing phase
        job_statuses[job_id].status = ProcessingStatus.AI_PROCESSING
        processed_files = await process_documents_parallel(job_id, request.openai_api_key, request.openai_model, request.max_concurrent)
        
        if not processed_files:
            raise Exception("No documents were successfully processed by OpenAI")

        job_statuses[job_id].status = ProcessingStatus.AI_COMPLETE
        job_statuses[job_id].message = f"AI processing completed. {len(processed_files)} files processed."

        # Combining phase
        job_statuses[job_id].status = ProcessingStatus.COMBINING
        final_file = await combine_documents(job_id)
        
        job_statuses[job_id].status = ProcessingStatus.COMPLETED
        job_statuses[job_id].completed_at = datetime.now()
        job_statuses[job_id].message = f"Processing completed. Output file: {final_file}"
            
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        job_statuses[job_id].status = ProcessingStatus.FAILED
        job_statuses[job_id].message = str(e)
        job_statuses[job_id].completed_at = datetime.now()

@app.post("/crawl", response_model=JobStatus)
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    job_id = hashlib.md5(f"{request.sitemap_url}_{datetime.now()}".encode()).hexdigest()
    
    job_status = JobStatus(
        job_id=job_id,
        status=ProcessingStatus.PENDING,
        created_at=datetime.now(),
        failed_urls=[]
    )
    
    job_statuses[job_id] = job_status
    background_tasks.add_task(process_crawl_request, job_id, request)
    return job_status

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_statuses[job_id]

@app.get("/download/{job_id}")
async def download_combined_document(job_id: str):
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_statuses[job_id].status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    filepath = os.path.join(COMBINED_DIR, f"combined_{job_id}.md")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Combined file not found")
    
    return FileResponse(filepath, filename=f"combined_{job_id}.md")

@app.get("/", response_class=HTMLResponse)
async def get_gui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)