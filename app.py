import telebot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
import logging
import re
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Flask для Render
from flask import Flask

# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==================== КОНФИГУРАЦИЯ ====================
# Токен берётся из переменной окружения Render (безопасно!)
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан в переменных окружения")

ADMIN_IDS = [int(x.strip()) for x in os.environ.get('ADMIN_IDS', '1812619337').split(',')]
PAYMENT_LINK = os.environ.get('PAYMENT_LINK', 'https://www.sberbank.ru/')

# ==================== СОСТОЯНИЯ ====================
STATES = {
    "MAIN_MENU": 0,
    "AWAITING_EXCEL": 1,
    "MANUAL_REVENUE": 3,
    "MANUAL_EXPENSES": 4,
    "MANUAL_DEBT": 5,
    "MANUAL_DAYS": 6,
    "MANUAL_CONFIRM": 7,
    "AWAITING_PAYMENT_PROOF": 8,
    "ADMIN_CONTACT": 9,
    "PAYMENT_DETAILS": 10,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

users_db = {}
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Flask приложение для Render
flask_app = Flask(__name__)

@flask_app.route('/')
def health():
    return "ProfitBot is running!", 200

@flask_app.route('/health')
def health_check():
    return "OK", 200

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def get_user(user_id):
    if user_id not in users_db:
        users_db[user_id] = {
            "free_used": False,
            "paid": False,
            "awaiting_payment": False,
            "temp_data": {},
            "state": STATES["MAIN_MENU"],
            "last_df": None
        }
    return users_db[user_id]

def set_state(user_id, state):
    get_user(user_id)["state"] = state

def is_admin(user_id):
    return user_id in ADMIN_IDS

def can_generate_report(user_id, user):
    if is_admin(user_id):
        return True, None
    if not user["free_used"]:
        return True, None
    if user.get("paid", False):
        return True, None
    return False, "❌ Это платный отчёт (99 ₽). Нажмите «💳 Оплатить»."

def parse_number(text):
    if not text:
        return None
    text = text.replace(' ', '')
    if ',' in text:
        parts = text.split(',')
        if len(parts) > 1 and len(parts[-1]) <= 2 and parts[-1].isdigit():
            text = text.replace(',', '.')
        else:
            text = text.replace(',', '')
    try:
        return float(text)
    except:
        return None

# ==================== ВСЕ ОСТАЛЬНЫЕ ФУНКЦИИ (AIAnalyzer, calculate_metrics, create_charts, generate_pdf_report, generate_excel_report, handle_text, handle_excel и т.д.) ====================
# ⚠️ СЮДА ВСТАВЬТЕ ВЕСЬ ОСТАЛЬНОЙ КОД ИЗ ВАШЕЙ РАБОЧЕЙ ВЕРСИИ (AIAnalyzer, calculate_metrics, create_charts, generate_pdf_report, generate_excel_report, клавиатуры, обработчики)
# Но без строки bot.polling() в конце — её убираем, так как бот будет запущен в потоке.

# ==================== ЗАПУСК ====================
def run_bot():
    """Запускает бота в отдельном потоке"""
    print("✅ ProfitBot запущен и слушает сообщения!")
    bot.infinity_polling()

if __name__ == "__main__":
    # Запускаем бота в фоновом потоке
    import threading
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Запускаем Flask для Render
    port = int(os.environ.get("PORT", 5000))
    print(f"Flask сервер запущен на порту {port}")
    flask_app.run(host="0.0.0.0", port=port)