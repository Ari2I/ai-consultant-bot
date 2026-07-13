# Образ на основе Debian slim — sentence-transformers и torch (его
# зависимость) не требуют системных библиотек сверх обычных, в
# отличие от Playwright/Chromium в прошлом проекте, поэтому базовый
# python-slim образ достаточен.
FROM python:3.11-slim

WORKDIR /app

# Устанавливаем Python-зависимости отдельным слоем — если
# requirements.txt не менялся, Docker переиспользует кэш и не
# переустанавливает всё заново (а зависимости здесь тяжёлые —
# sentence-transformers тянет torch).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения.
COPY . .

# Контейнер не должен работать от root без необходимости — создаём
# отдельного пользователя и передаём ему права на рабочий каталог
# (включая кэш модели эмбеддингов, который появится при первом
# запуске в $HOME/.cache/huggingface).
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
