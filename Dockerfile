FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Run as non-root
RUN useradd --create-home --uid 1000 trainlocks

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# rapidocr also installs the GUI opencv build (libGL-linked); headless is the
# same cv2 module without it. Remove the GUI leftovers so the runtime image
# needs no libGL, then pre-download the OCR models so the first request works
# fully offline.
RUN rm -rf /app/.venv/lib/python3*/site-packages/opencv_python.libs \
           /app/.venv/lib/python3*/site-packages/opencv_python-*.dist-info \
    && /app/.venv/bin/python - <<'EOF'
from rapidocr import RapidOCR, LangRec, ModelType, OCRVersion
RapidOCR(params={
    "Rec.ocr_version": OCRVersion.PPOCRV5,
    "Rec.model_type": ModelType.MOBILE,
    "Rec.lang_type": LangRec.EN,
})
print("OCR models pre-downloaded")
EOF

COPY app/ ./app/

VOLUME ["/data"]
EXPOSE 8000

USER trainlocks

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]