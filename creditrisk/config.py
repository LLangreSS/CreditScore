"""Все константы проекта в одном месте."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cs-training.csv"  # Kaggle: Give Me Some Credit
# Публичные зеркала датасета. Не гарантированы — основной путь: положить файл в data/.
DATA_URLS: tuple[str, ...] = tuple(
    u for u in (
        os.getenv("CREDIT_DATA_URL"),
        "https://raw.githubusercontent.com/selva86/datasets/master/cs-training.csv",
    ) if u
)

RANDOM_STATE = 42
SAMPLE_LIMIT = 50_000          # лимит выборки для быстрого UI (стратифицированный)
TARGET = "default"

AGE_MIN, AGE_MAX = 18, 95
AGE_BINS = [18, 25, 35, 50, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-50", "51-65", "65+"]
BKI_ERROR_CODE = 96            # в сырых данных 96 и 98 — технические коды БКИ, не количество просрочек

# Веса бакетов просрочки в сводном индексе тяжести
DELINQ_WEIGHTS = {"d30": 1.0, "d60": 2.0, "d90": 3.5}

FEATURES = [
    "credit_utilization", "age", "dti_ratio", "monthly_income", "income_per_person",
    "is_income_missing", "open_credit_lines", "real_estate_loans", "dependents",
    "delinq_30_59_days", "delinq_60_89_days", "delinq_90_plus_days", "weighted_delinq_score",
]

# Шкала скорингового балла (аналог кредитного рейтинга; не FICO — это торговая марка)
SCORE_BASE, SCORE_BASE_PD, SCORE_PDO = 600, 0.05, 50
SCORE_MIN, SCORE_MAX = 300, 850

# Предположения для финансовой симуляции (в датасете нет суммы кредита — это оценка!)
LOAN_INCOME_MULT = 0.45
LOAN_MIN, LOAN_MAX = 1_500, 150_000
