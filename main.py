from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from mapper_404 import match_404s
import pandas as pd
import tempfile
import traceback
import os
import uuid

app = FastAPI()

# Store output files in a temp directory
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "url_mapper_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/map404s")
async def map_404s(broken_csv: UploadFile, live_csv: UploadFile):
    try:
        # Save uploaded CSVs
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f2:
            f1.write(await broken_csv.read())
            f2.write(await live_csv.read())

            # Create a unique output filename
            output_path = os.path.join(OUTPUT_DIR, f"results_{uuid.uuid4().hex}.csv")

            # Run the mapping
            result_df = match_404s(f1.name, f2.name, output_csv=output_path)

            if result_df is None or result_df.empty:
                raise HTTPException(status_code=500, detail="No matches returned.")

            # Build a link to download the results
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

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serve the generated CSV for download."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="text/csv", filename=filename)
