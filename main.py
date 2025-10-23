from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mapper_404 import match_404s
import pandas as pd
import tempfile
import traceback
import os
import uuid

# ---------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------
app = FastAPI(title="404 Redirect Mapper", version="2.0")

# Allow cross-origin requests so your whole team can use the tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary output directory
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "url_mapper_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Homepage route
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple landing page with link to /docs."""
    return """
    <html>
      <head>
        <title>404 Redirect Mapper</title>
        <style>
          body { font-family: sans-serif; margin: 3em; background: #fafafa; color: #333; }
          h2 { color: #222; }
          a.button {
            display: inline-block;
            background: #0084ff;
            color: white;
            padding: 12px 22px;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
          }
          a.button:hover { background: #006edc; }
          .container {
            max-width: 700px;
            margin: 0 auto;
            text-align: center;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h2>🧭 404 Redirect Mapper</h2>
          <p>This tool maps broken URLs to the most relevant live pages using OpenAI embeddings.</p>
          <p><a href="/docs" class="button">Open Tool Interface</a></p>
          <p style="margin-top: 1.5em; font-size: 0.9em; color: #666;">Version 2.0 — Cloud hosted via Render</p>
        </div>
      </body>
    </html>
    """

# ---------------------------------------------------------------------
# Main mapping endpoint
# ---------------------------------------------------------------------
@app.post("/map404s")
async def map_404s(broken_csv: UploadFile, live_csv: UploadFile):
    """Accept two CSVs (broken + live) and return redirect matches."""
    try:
        # Save uploaded CSVs to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f2:
            f1.write(await broken_csv.read())
            f2.write(await live_csv.read())

            # Create unique output filename
            output_path = os.path.join(OUTPUT_DIR, f"results_{uuid.uuid4().hex}.csv")

            # Run matching function
            result_df = match_404s(f1.name, f2.name, output_csv=output_path)

            if result_df is None or result_df.empty:
                raise HTTPException(status_code=500, detail="No matches returned.")

            download_url = f"/download/{os.path.basename(output_path)}"
            message = {
                "message": "✅ Matching complete!",
                "download_link": download_url,
                "rows_returned": len(result_df)
            }

            return JSONResponse(content=message)

    except Exception as e:
        print("❌ Error in /map404s endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------
# Download endpoint
# ---------------------------------------------------------------------
@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serve the generated CSV for download."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="text/csv", filename="redirect_map.csv")
