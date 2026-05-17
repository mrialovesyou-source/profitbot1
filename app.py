# app.py — финальная версия для Render (webhook, всё включено)

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

# Flask и webhook
from flask import Flask, request

# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==================== КОНФИГУРАЦИЯ ====================
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

# Flask app
flask_app = Flask(__name__)

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

# ==================== AI-АНАЛИТИКА ====================
class AIAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        if 'Дата' in self.df.columns:
            self.df['Дата'] = pd.to_datetime(self.df['Дата'])
            self.df.sort_values('Дата', inplace=True)
        self.df['Чистый_поток'] = self.df['Выручка'] - self.df['Расходы']
        self.df['Накопленный_поток'] = self.df['Чистый_поток'].cumsum()

    def predict_cash_gap(self, forecast_days=14):
        if len(self.df) < 7:
            return None, "Недостаточно данных (минимум 7 дней)"
        y = self.df['Чистый_поток'].values
        dates = self.df['Дата']
        best_aic = np.inf
        best_order = (1,0,1)
        for p in range(0,3):
            for q in range(0,3):
                try:
                    model = ARIMA(y, order=(p,0,q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p,0,q)
                except:
                    continue
        try:
            model = ARIMA(y, order=best_order)
            fit = model.fit()
            forecast = fit.forecast(steps=forecast_days)
        except:
            model_es = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=7)
            fit_es = model_es.fit()
            forecast = fit_es.forecast(forecast_days)
        X = np.arange(len(self.df)).reshape(-1,1)
        y_cum = self.df['Накопленный_поток'].values
        lr = LinearRegression()
        lr.fit(X, y_cum)
        future_X = np.arange(len(self.df), len(self.df)+forecast_days).reshape(-1,1)
        cum_forecast = lr.predict(future_X)
        gap_day = next((i for i, val in enumerate(cum_forecast) if val < 0), None)
        forecast_dates = [dates.iloc[-1] + timedelta(days=i+1) for i in range(forecast_days)]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(dates, y, 'o-', label='Факт')
        ax.plot(forecast_dates, forecast, 'r--', label='Прогноз')
        ax.axhline(0, color='k', linestyle='-', alpha=0.5)
        ax.set_title('Прогноз чистого денежного потока')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Рубли')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        if gap_day:
            gap_date = dates.iloc[0] + timedelta(days=gap_day) if gap_day < len(dates) else forecast_dates[gap_day - len(dates)]
            conclusion = f"⚠️ Ожидается кассовый разрыв около {gap_date.strftime('%d.%m.%Y')}"
        else:
            conclusion = "✅ Кассовых разрывов не ожидается"
        return fig, conclusion

    def detect_fraud(self, z_threshold=3.0):
        if len(self.df) < 5:
            return pd.DataFrame(), None, "Недостаточно данных"
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        features = self.df[['Расходы']].copy()
        features['day_index'] = np.arange(len(features))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        preds = iso_forest.fit_predict(scaled)
        outliers = preds == -1
        z_scores = np.abs((self.df['Расходы'] - self.df['Расходы'].mean()) / self.df['Расходы'].std())
        z_outliers = z_scores > z_threshold
        combined = outliers | z_outliers
        if not combined.any():
            return pd.DataFrame(), None, "Аномалий не обнаружено"
        fraud_df = self.df[combined][['Дата', 'Расходы']].copy()
        fraud_df.columns = ['Дата', 'Сумма']
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(self.df['Дата'], self.df['Расходы'], alpha=0.5, label='Норма')
        ax.scatter(fraud_df['Дата'], fraud_df['Сумма'], color='red', s=80, label='Аномалии')
        ax.set_title('Аномалии в расходах')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Сумма, руб')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        conclusion = f"Обнаружено {len(fraud_df)} аномальных операций"
        return fraud_df, fig, conclusion

    def suggest_investment_strategy(self):
        total_rev = self.df['Выручка'].sum()
        total_exp = self.df['Расходы'].sum()
        net_profit = total_rev - total_exp
        roi = (net_profit / total_exp) * 100 if total_exp != 0 else 0
        reinvest_pct = 20 if roi < 15 else (30 if roi < 30 else 40)
        conclusion = f"Рекомендуется реинвестировать {reinvest_pct}% от чистой прибыли ({net_profit*reinvest_pct/100:.0f} руб)"
        return pd.DataFrame(), None, conclusion

    def budget_optimization(self, forecast_days=30):
        if len(self.df) < 7:
            return None, "Недостаточно данных"
        rev_series = self.df['Выручка'].values
        exp_series = self.df['Расходы'].values
        try:
            model_rev = ExponentialSmoothing(rev_series, trend='add', seasonal='add', seasonal_periods=7)
            fit_rev = model_rev.fit()
            rev_forecast = fit_rev.forecast(forecast_days)
        except:
            rev_forecast = np.full(forecast_days, self.df['Выручка'].mean())
        try:
            model_exp = ExponentialSmoothing(exp_series, trend='add', seasonal='add', seasonal_periods=7)
            fit_exp = model_exp.fit()
            exp_forecast = fit_exp.forecast(forecast_days)
        except:
            exp_forecast = np.full(forecast_days, self.df['Расходы'].mean())
        last_dates = self.df['Дата'].tail(14)
        forecast_dates = [last_dates.iloc[-1] + timedelta(days=i+1) for i in range(forecast_days)]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(self.df['Дата'].tail(14), self.df['Выручка'].tail(14), 'b-o', label='Факт выручка')
        ax.plot(self.df['Дата'].tail(14), self.df['Расходы'].tail(14), 'r-s', label='Факт расходы')
        ax.plot(forecast_dates, rev_forecast, 'b--', label='Прогноз выручки')
        ax.plot(forecast_dates, exp_forecast, 'r--', label='Прогноз расходов')
        ax.set_title('Бюджетирование: План vs Факт')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Рубли')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        avg_rev = np.mean(rev_forecast)
        recommended_budget = avg_rev * 0.7
        budget_rec = {
            'Рекомендуемый бюджет расходов': recommended_budget,
            'Совет': 'Бюджет в пределах нормы' if recommended_budget > np.mean(exp_forecast) else 'Необходимо сократить расходы'
        }
        return fig, budget_rec

# ==================== РАСЧЁТ МЕТРИК ====================
def calculate_metrics(df):
    total_rev = df['Выручка'].sum()
    total_exp = df['Расходы'].sum()
    days = (df['Дата'].max() - df['Дата'].min()).days + 1 if len(df) > 0 else 30
    avg_daily = total_rev / days
    profit = total_rev - total_exp
    roi = (profit / total_exp) * 100 if total_exp != 0 else 0
    breakeven = avg_daily * 0.7
    safety = ((total_rev - (breakeven * days)) / total_rev) * 100 if total_rev != 0 else 0
    forecast = avg_daily * 7
    if roi < 15:
        roi_comment = f"⚠️ Рентабельность низкая ({roi:.1f}%). Оптимизируйте затраты."
    elif roi < 30:
        roi_comment = f"📈 Рентабельность средняя ({roi:.1f}%). Хорошо."
    else:
        roi_comment = f"🔥 Рентабельность отличная ({roi:.1f}%). Масштабируйте!"
    if safety < 20:
        margin_comment = f"⚠️ Запас прочности низкий ({safety:.1f}%). Создайте резерв."
    elif safety < 50:
        margin_comment = f"✅ Запас прочности достаточный ({safety:.1f}%)."
    else:
        margin_comment = f"🛡️ Запас прочности высокий ({safety:.1f}%)."
    if avg_daily < breakeven:
        be_comment = f"🔴 КРИТИЧЕСКАЯ СИТУАЦИЯ: выручка ({avg_daily:.0f}₽/день) ниже точки безубыточности ({breakeven:.0f}₽/день)."
    else:
        be_comment = f"✅ Выручка выше точки безубыточности."
    if profit > 0:
        reinvest_comment = f"💡 Реинвестируйте 30% прибыли ({int(profit*0.3)}₽) в развитие."
    else:
        reinvest_comment = "⚠️ Бизнес убыточен. Пересмотрите расходы."
    return {
        'total_revenue': total_rev, 'total_expenses': total_exp, 'days': days,
        'avg_daily_revenue': avg_daily, 'net_profit': profit, 'roi': roi,
        'breakeven_point': breakeven, 'safety_margin': safety, 'forecast': forecast,
        'roi_comment': roi_comment, 'margin_comment': margin_comment,
        'be_comment': be_comment, 'reinvest_comment': reinvest_comment
    }

# ==================== ГРАФИКИ ====================
def create_charts(df, metrics):
    temp_dir = tempfile.mkdtemp()
    charts = {}
    if len(df) >= 3:
        fig1, ax1 = plt.subplots(figsize=(10,5))
        ax1.plot(df['Дата'], df['Выручка'], 'b-o', label='Выручка')
        ax1.plot(df['Дата'], df['Расходы'], 'r-s', label='Расходы')
        ax1.set_title('Динамика выручки и расходов')
        ax1.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        p1 = os.path.join(temp_dir, 'chart1.png')
        plt.savefig(p1, dpi=100, bbox_inches='tight')
        plt.close()
        charts['dynamics'] = p1

        df['Прибыль'] = df['Выручка'] - df['Расходы']
        fig2, ax2 = plt.subplots(figsize=(10,5))
        colors = ['green' if x>=0 else 'red' for x in df['Прибыль']]
        ax2.bar(df['Дата'], df['Прибыль'], color=colors)
        ax2.axhline(0, color='black')
        ax2.set_title('Прибыль/убыток по дням')
        plt.xticks(rotation=45)
        plt.tight_layout()
        p2 = os.path.join(temp_dir, 'chart2.png')
        plt.savefig(p2, dpi=100, bbox_inches='tight')
        plt.close()
        charts['profit'] = p2

    fig3, ax3 = plt.subplots(figsize=(8,5))
    names = ['ROI (%)', 'Запас прочности (%)']
    vals = [metrics['roi'], metrics['safety_margin']]
    colors = ['#2E7D32' if v>=20 else '#FF6B6B' for v in vals]
    ax3.barh(names, vals, color=colors)
    ax3.axvline(15, color='orange', ls='--', label='Порог ROI')
    ax3.axvline(20, color='blue', ls='--', label='Порог запаса')
    ax3.legend()
    ax3.set_title('Ключевые метрики')
    plt.tight_layout()
    p3 = os.path.join(temp_dir, 'chart3.png')
    plt.savefig(p3, dpi=100, bbox_inches='tight')
    plt.close()
    charts['metrics'] = p3

    if 'Категория' in df.columns:
        exp_cat = df.groupby('Категория')['Расходы'].sum()
        if len(exp_cat) > 1:
            fig4, ax4 = plt.subplots(figsize=(8,6))
            ax4.pie(exp_cat.values, labels=exp_cat.index, autopct='%1.1f%%')
            ax4.set_title('Структура расходов')
            plt.tight_layout()
            p4 = os.path.join(temp_dir, 'chart4.png')
            plt.savefig(p4, dpi=100, bbox_inches='tight')
            plt.close()
            charts['expenses_pie'] = p4
    return charts, temp_dir

# ==================== PDF И EXCEL ====================
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import textwrap

def generate_pdf_report(metrics, df, charts):
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, 'report.pdf')
    try:
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        if not os.path.exists(font_path):
            font_path = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DejaVu', font_path))
            font_name = 'DejaVu'
        else:
            font_name = 'Helvetica'
    except:
        font_name = 'Helvetica'
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 35
    def draw_image(img_path, y_pos, w=500, h=170):
        if not img_path or not os.path.exists(img_path):
            return y_pos - 20
        c.drawImage(ImageReader(img_path), 40, y_pos - h, width=w, height=h, preserveAspectRatio=True)
        return y_pos - h - 15

    c.setFont(font_name, 18)
    c.drawString(30, y, "ProfitBot - Аналитический отчёт")
    y -= 28
    c.setFont(font_name, 10)
    c.drawString(30, y, f"Дата отчёта: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    y -= 18
    c.drawString(30, y, f"Анализируемый период: {metrics['days']} дней")
    y -= 25

    # 1. Главный вывод
    c.setFont(font_name, 12)
    c.setFillColorRGB(0.18,0.49,0.20)
    c.drawString(30, y, "1. Главный вывод")
    c.setFillColorRGB(0,0,0)
    y -= 20
    if metrics['net_profit'] > 0:
        main_conclusion = f"Ваш бизнес ПРИБЫЛЬНЫЙ. Чистая прибыль составляет {metrics['net_profit']:,.2f} рублей за {metrics['days']} дней. Вы зарабатываете больше, чем тратите."
    else:
        main_conclusion = f"Ваш бизнес УБЫТОЧНЫЙ. Убыток составляет {abs(metrics['net_profit']):,.2f} рублей за {metrics['days']} дней. Вы тратите больше, чем зарабатываете."
    c.setFont(font_name, 10)
    for line in textwrap.wrap(main_conclusion, width=95):
        c.drawString(35, y, line)
        y -= 15
    y -= 10

    # 2. Что означают цифры
    if y < 200:
        c.showPage()
        y = height - 35
    c.setFont(font_name, 12)
    c.setFillColorRGB(0.18,0.49,0.20)
    c.drawString(30, y, "2. Что означают цифры")
    c.setFillColorRGB(0,0,0)
    y -= 18
    definitions = [
        ("Выручка за период", f"{metrics['total_revenue']:,.2f} руб", "Все деньги, которые поступили на счёт за указанный период."),
        ("Расходы за период", f"{metrics['total_expenses']:,.2f} руб", "Все затраты: аренда, закупки, зарплаты, налоги, реклама."),
        ("Чистая прибыль", f"{metrics['net_profit']:,.2f} руб", "Выручка минус расходы. Если число отрицательное - бизнес в убытке."),
        ("ROI (рентабельность)", f"{metrics['roi']:.1f}%", "Показывает эффективность бизнеса. Формула: (Прибыль / Расходы) * 100%."),
        ("Точка безубыточности", f"{metrics['breakeven_point']:.2f} руб/день", "Минимальная выручка для покрытия расходов в день."),
        ("Запас прочности", f"{metrics['safety_margin']:.1f}%", "На сколько процентов может упасть выручка без убытков.")
    ]
    c.setFont(font_name, 9)
    for name, value, explanation in definitions:
        if y < 100:
            c.showPage()
            y = height - 35
        c.drawString(35, y, f"- {name}: {value}")
        y -= 13
        c.setFont(font_name, 8)
        c.setFillColorRGB(0.4,0.4,0.4)
        for line in textwrap.wrap(explanation, width=90):
            c.drawString(45, y, line)
            y -= 12
        c.setFillColorRGB(0,0,0)
        y -= 5
    y -= 5

    # 3. Детальный анализ
    if y < 250:
        c.showPage()
        y = height - 35
    c.setFont(font_name, 12)
    c.setFillColorRGB(0.18,0.49,0.20)
    c.drawString(30, y, "3. Детальный анализ")
    c.setFillColorRGB(0,0,0)
    y -= 18
    c.setFont(font_name, 9)
    if metrics['roi'] < 0:
        analysis1 = f"- Рентабельность отрицательная ({metrics['roi']:.1f}%). Каждый рубль расходов приносит убыток."
    elif metrics['roi'] < 15:
        analysis1 = f"- Рентабельность низкая ({metrics['roi']:.1f}%). Эффективность ниже нормы."
    elif metrics['roi'] < 30:
        analysis1 = f"- Рентабельность средняя ({metrics['roi']:.1f}%). Хороший результат."
    else:
        analysis1 = f"- Рентабельность высокая ({metrics['roi']:.1f}%)! Очень эффективный бизнес."
    for line in textwrap.wrap(analysis1, width=95):
        c.drawString(35, y, line)
        y -= 14
    y -= 5
    if metrics['safety_margin'] < 20:
        analysis2 = f"- Запас прочности низкий ({metrics['safety_margin']:.1f}%). Даже небольшое падение выручки приведёт к убыткам."
    elif metrics['safety_margin'] < 50:
        analysis2 = f"- Запас прочности достаточный ({metrics['safety_margin']:.1f}%). Бизнес устойчив."
    else:
        analysis2 = f"- Запас прочности высокий ({metrics['safety_margin']:.1f}%)! Бизнес очень устойчив."
    for line in textwrap.wrap(analysis2, width=95):
        c.drawString(35, y, line)
        y -= 14
    y -= 5
    if metrics['avg_daily_revenue'] < metrics['breakeven_point']:
        analysis3 = f"- КРИТИЧЕСКИ: ежедневная выручка ({metrics['avg_daily_revenue']:.0f} руб) НИЖЕ точки безубыточности ({metrics['breakeven_point']:.0f} руб/день)"
    else:
        analysis3 = f"- Ежедневная выручка ({metrics['avg_daily_revenue']:.0f} руб) ВЫШЕ точки безубыточности ({metrics['breakeven_point']:.0f} руб/день)"
    for line in textwrap.wrap(analysis3, width=95):
        c.drawString(35, y, line)
        y -= 14
    y -= 5
    analysis4 = f"- Прогноз выручки на следующую неделю: {metrics['forecast']:,.0f} руб"
    for line in textwrap.wrap(analysis4, width=95):
        c.drawString(35, y, line)
        y -= 14
    y -= 10

    # 4-7. Графики
    if 'dynamics' in charts:
        if y < 200:
            c.showPage()
            y = height - 35
        else:
            y -= 10
        c.setFont(font_name, 12)
        c.drawString(30, y, "4. График 1: Динамика выручки и расходов")
        y -= 18
        c.setFont(font_name, 8)
        c.setFillColorRGB(0.4,0.4,0.4)
        c.drawString(35, y, "(Синяя линия - выручка, красная - расходы. Помогает сравнить доходы и расходы во времени)")
        c.setFillColorRGB(0,0,0)
        y -= 15
        y = draw_image(charts['dynamics'], y, w=520, h=160)
        y -= 10

    if 'profit' in charts:
        if y < 220:
            c.showPage()
            y = height - 35
        else:
            y -= 5
        c.setFont(font_name, 12)
        c.drawString(30, y, "5. График 2: Прибыль/убыток по дням")
        y -= 18
        c.setFont(font_name, 8)
        c.setFillColorRGB(0.4,0.4,0.4)
        c.drawString(35, y, "(Зелёные столбцы - прибыль, красные - убыток. Показывает, какие дни были прибыльными)")
        c.setFillColorRGB(0,0,0)
        y -= 15
        y = draw_image(charts['profit'], y, w=520, h=160)
        y -= 10

    if 'metrics' in charts:
        if y < 250:
            c.showPage()
            y = height - 35
        c.setFont(font_name, 12)
        c.drawString(30, y, "6. График 3: Ключевые метрики")
        y -= 18
        c.setFont(font_name, 8)
        c.setFillColorRGB(0.4,0.4,0.4)
        c.drawString(35, y, "(Сравнивает ROI и запас прочности с нормативами 15% и 20%)")
        c.setFillColorRGB(0,0,0)
        y -= 15
        y = draw_image(charts['metrics'], y, w=450, h=160)
        y -= 10

    if 'expenses_pie' in charts:
        if y < 280:
            c.showPage()
            y = height - 35
        c.setFont(font_name, 12)
        c.drawString(30, y, "7. График 4: Структура расходов")
        y -= 18
        c.setFont(font_name, 8)
        c.setFillColorRGB(0.4,0.4,0.4)
        c.drawString(35, y, "(Показывает, на что уходит больше всего денег. Помогает найти точки оптимизации)")
        c.setFillColorRGB(0,0,0)
        y -= 15
        y = draw_image(charts['expenses_pie'], y, w=400, h=240)
        y -= 10

    # 8. Рекомендации
    if y < 250:
        c.showPage()
        y = height - 35
    else:
        y -= 10
    c.setFont(font_name, 12)
    c.setFillColorRGB(0.18,0.49,0.20)
    c.drawString(30, y, "8. Рекомендации на основе ваших данных")
    c.setFillColorRGB(0,0,0)
    y -= 20
    recommendations = []
    if metrics['net_profit'] > 0:
        reinvest = int(metrics['net_profit'] * 0.3)
        recommendations.append(f"✅ Ваш бизнес прибыльный. Чистая прибыль составила {metrics['net_profit']:,.0f} рублей. Рекомендуется реинвестировать {reinvest:,} рублей (30%) в развитие: маркетинг, новые товары, обучение персонала.")
    else:
        recommendations.append(f"⚠️ Бизнес убыточный. Убыток составил {abs(metrics['net_profit']):,.0f} рублей. Необходимо срочно: создать план увеличения выручки или сокращения расходов.")
    if metrics['roi'] < 15:
        recommendations.append(f"📉 Рентабельность низкая ({metrics['roi']:.1f}%). Проведите аудит расходов. Рассмотрите повышение цен на 5-10%.")
    elif metrics['roi'] < 30:
        recommendations.append(f"📈 Рентабельность средняя ({metrics['roi']:.1f}%). Хороший результат, есть потенциал для роста. Увеличьте маркетинговый бюджет.")
    elif metrics['roi'] < 100:
        recommendations.append(f"💰 Рентабельность хорошая ({metrics['roi']:.1f}%). Масштабируйте успешные каналы, расширяйте ассортимент.")
    else:
        recommendations.append(f"🔥 Рентабельность отличная ({metrics['roi']:.1f}%)! Рекомендуется агрессивное масштабирование.")
    if metrics['safety_margin'] < 20:
        recommendations.append(f"🛡️ Запас прочности низкий ({metrics['safety_margin']:.1f}%). Создайте финансовую подушку безопасности (10% от выручки в резерв).")
    elif metrics['safety_margin'] < 50:
        recommendations.append(f"✅ Запас прочности достаточный ({metrics['safety_margin']:.1f}%). Бизнес выдержит падение выручки до {metrics['safety_margin']:.0f}% без убытков.")
    else:
        recommendations.append(f"🛡️ Запас прочности высокий ({metrics['safety_margin']:.1f}%)! Бизнес очень устойчив к кризисам.")
    if metrics['avg_daily_revenue'] < metrics['breakeven_point']:
        recommendations.append(f"🔴 КРИТИЧЕСКИ: ваша выручка ({metrics['avg_daily_revenue']:.0f}₽/день) НИЖЕ точки безубыточности ({metrics['breakeven_point']:.0f}₽/день). Срочно увеличивайте продажи!")
    else:
        recommendations.append(f"✅ Ваша выручка выше точки безубыточности. Каждый день приносит прибыль. Увеличьте выручку ещё на 20-30%.")
    if metrics['days'] >= 30:
        recommendations.append(f"📊 Анализ проведён за {metrics['days']} дней. Данные достоверны. Проводите такой анализ раз в месяц.")
    else:
        recommendations.append(f"⚠️ Период анализа мал ({metrics['days']} дней). Накопите данные за 30+ дней для точных выводов.")
    daily_profit = metrics['net_profit'] / metrics['days']
    if daily_profit < 0:
        recommendations.append(f"📉 Среднедневная прибыль отрицательная ({daily_profit:.0f}₽/день). Нужно срочно менять стратегию.")
    elif daily_profit < 100:
        recommendations.append(f"💵 Среднедневная прибыль небольшая ({daily_profit:.0f}₽/день). Рассмотрите расширение ассортимента.")
    else:
        recommendations.append(f"💰 Среднедневная прибыль {daily_profit:.0f}₽/день. Хорошо. Для роста увеличьте маркетинговую активность.")
    c.setFont(font_name, 9)
    for rec in recommendations:
        for line in textwrap.wrap(rec, width=95):
            c.drawString(35, y, line)
            y -= 14
        y -= 5
    c.setFont(font_name, 8)
    c.setFillColorRGB(0.5,0.5,0.5)
    c.drawString(30, 30, f"© ProfitBot - ваш финансовый помощник | {datetime.now().strftime('%d.%m.%Y')}")
    c.save()
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    shutil.rmtree(temp_dir)
    return pdf_bytes

def generate_excel_report(df, metrics, manual_mode=False):
    output = BytesIO()
    df_copy = df.copy()
    if 'Дата' in df_copy.columns:
        df_copy['Дата'] = pd.to_datetime(df_copy['Дата']).dt.strftime('%d.%m.%Y')
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if manual_mode or df is None:
            data = {
                'Показатель': ['Выручка', 'Расходы', 'Долги', 'Период (дней)'],
                'Значение': [metrics['total_revenue'], metrics['total_expenses'], 0, metrics['days']]
            }
            pd.DataFrame(data).to_excel(writer, sheet_name='Исходные данные', index=False)
        else:
            df_copy.to_excel(writer, sheet_name='Исходные данные', index=False)
        calc_data = {
            'Показатель': ['Среднедневная выручка', 'ROI (%)', 'Чистая прибыль', 'Запас прочности (%)', 'Точка безубыточности', 'Прогноз на неделю'],
            'Значение': [metrics['avg_daily_revenue'], metrics['roi'], metrics['net_profit'], metrics['safety_margin'], metrics['breakeven_point'], metrics['forecast']]
        }
        pd.DataFrame(calc_data).to_excel(writer, sheet_name='Расчёты', index=False)
        insights = pd.DataFrame({
            'Вывод': [metrics['roi_comment'], metrics['margin_comment'], metrics['be_comment'], metrics['reinvest_comment']]
        })
        insights.to_excel(writer, sheet_name='Аналитика', index=False)
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_len = 0
                col_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_len:
                            max_len = len(str(cell.value))
                    except:
                        pass
                worksheet.column_dimensions[col_letter].width = min(max_len+2, 40)
    output.seek(0)
    return output

# ==================== КЛАВИАТУРЫ И ОБРАБОТЧИКИ ====================
def main_menu_keyboard():
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row("📥 Загрузить Excel", "✍️ Ввести вручную")
    markup.row("📞 Связь с админом", "💳 Детали оплаты")
    return markup

def back_button():
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row("◀️ Назад")
    return markup

@bot.message_handler(commands=['start'])
def start_cmd(message):
    user_id = message.chat.id
    get_user(user_id)
    set_state(user_id, STATES["MAIN_MENU"])
    welcome_text = (
        "🤖 *ProfitBot – ваш финансовый помощник в Telegram!*\n\n"
        "📊 *Что я умею:*\n"
        "✅ Анализировать Excel-файлы (выручка, расходы, даты)\n"
        "✍️ Принимать данные вручную (по шагам)\n"
        "📈 Строить графики динамики продаж, прибыли и метрик\n"
        "🔮 Прогнозировать кассовые разрывы с помощью ИИ\n"
        "🕵️ Выявлять аномалии в расходах\n"
        "💰 Рассчитывать ROI, точку безубыточности, запас прочности\n"
        "📄 Создавать подробные PDF- и Excel-отчёты\n\n"
        "🔒 *Первый отчёт бесплатно, далее – 99 ₽*\n\n"
        "👇 Просто выберите действие в меню ниже."
    )
    bot.send_message(user_id, welcome_text, reply_markup=main_menu_keyboard(), parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def help_cmd(message):
    help_text = (
        "📖 *Помощь по ProfitBot*\n\n"
        "🚀 /start – главное меню\n"
        "❓ /help – эта справка\n"
        "📞 Вопросы – админу через кнопку «Связь с админом»."
    )
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

def send_brief_summary(user_id, metrics):
    summary = (
        "📌 *Краткий итог по вашему бизнесу*\n\n"
        f"💰 *Выручка за период:* {metrics['total_revenue']:,.2f} ₽\n"
        f"📉 *Расходы:* {metrics['total_expenses']:,.2f} ₽\n"
        f"💵 *Чистая прибыль:* {metrics['net_profit']:,.2f} ₽\n\n"
        f"📊 *ROI (рентабельность):* {metrics['roi']:.1f}%\n"
        f"🛡️ *Запас прочности:* {metrics['safety_margin']:.1f}%\n"
        f"⚖️ *Точка безубыточности:* {metrics['breakeven_point']:.2f} ₽/день\n\n"
    )
    if metrics['net_profit'] > 0:
        summary += "✅ *Вывод:* Бизнес прибыльный. Хорошая работа!\n"
    else:
        summary += "⚠️ *Вывод:* Бизнес убыточен. Рекомендуется оптимизировать расходы.\n"
    summary += "\n📄 *Подробный отчёт с графиками и рекомендациями – в приложенном PDF.*"
    bot.send_message(user_id, summary, parse_mode='Markdown')

@bot.message_handler(func=lambda m: True, content_types=['text'])
def handle_text(message):
    user_id = message.chat.id
    text = message.text.strip()
    user = get_user(user_id)
    state = user["state"]

    if text == "◀️ Назад":
        set_state(user_id, STATES["MAIN_MENU"])
        bot.send_message(user_id, "🏠 Главное меню", reply_markup=main_menu_keyboard())
        return

    if state == STATES["MAIN_MENU"]:
        if text == "📥 Загрузить Excel":
            can, msg = can_generate_report(user_id, user)
            if not can:
                bot.send_message(user_id, msg)
                return
            set_state(user_id, STATES["AWAITING_EXCEL"])
            bot.send_message(user_id, "📁 Загрузите Excel-файл с колонками: Дата, Выручка, Расходы", reply_markup=back_button())
        elif text == "✍️ Ввести вручную":
            can, msg = can_generate_report(user_id, user)
            if not can:
                bot.send_message(user_id, msg)
                return
            user["temp_data"] = {}
            set_state(user_id, STATES["MANUAL_DAYS"])
            bot.send_message(user_id, "📅 За сколько дней данные? (например: 30):", reply_markup=back_button())
        elif text == "📞 Связь с админом":
            set_state(user_id, STATES["ADMIN_CONTACT"])
            bot.send_message(user_id, "📝 Напишите ваше сообщение администратору:", reply_markup=back_button())
        elif text == "💳 Детали оплаты":
            if is_admin(user_id):
                bot.send_message(user_id, "👑 Вы администратор – все отчёты бесплатны.")
                set_state(user_id, STATES["MAIN_MENU"])
                bot.send_message(user_id, "🏠 Главное меню", reply_markup=main_menu_keyboard())
                return
            set_state(user_id, STATES["PAYMENT_DETAILS"])
            bot.send_message(user_id, f"💳 *Оплата доступа к отчётам*\n\n🔹 Первый отчёт – *бесплатно*.\n🔹 Последующие – *99 ₽*.\n🔗 Ссылка: {PAYMENT_LINK}\n\n📸 После оплаты нажмите «📤 Отправить чек».", parse_mode='Markdown', reply_markup=back_button())
        else:
            bot.send_message(user_id, "❓ Используйте кнопки меню.", reply_markup=main_menu_keyboard())

    elif state == STATES["MANUAL_DAYS"]:
        try:
            user["temp_data"]["days"] = int(text)
            set_state(user_id, STATES["MANUAL_REVENUE"])
            bot.send_message(user_id, "💰 Введите выручку (руб):", reply_markup=back_button())
        except:
            bot.send_message(user_id, "❌ Введите целое число дней.", reply_markup=back_button())
    elif state == STATES["MANUAL_REVENUE"]:
        try:
            val = parse_number(text)
            if val is None:
                raise ValueError
            user["temp_data"]["revenue"] = val
            set_state(user_id, STATES["MANUAL_EXPENSES"])
            bot.send_message(user_id, "📉 Введите расходы (руб):", reply_markup=back_button())
        except:
            bot.send_message(user_id, "❌ Введите число (можно использовать пробелы или запятые, например 600 000 или 4000,6).", reply_markup=back_button())
    elif state == STATES["MANUAL_EXPENSES"]:
        try:
            val = parse_number(text)
            if val is None:
                raise ValueError
            user["temp_data"]["expenses"] = val
            set_state(user_id, STATES["MANUAL_DEBT"])
            bot.send_message(user_id, "🏦 Введите долги (руб):", reply_markup=back_button())
        except:
            bot.send_message(user_id, "❌ Введите число (можно использовать пробелы или запятые).", reply_markup=back_button())
    elif state == STATES["MANUAL_DEBT"]:
        try:
            val = parse_number(text)
            if val is None:
                raise ValueError
            user["temp_data"]["debt"] = val
            td = user["temp_data"]
            confirm = f"✅ *Проверьте данные:*\n\n"
            confirm += f"📅 Период: {td['days']} дней\n"
            confirm += f"💰 Выручка: {td['revenue']:.2f} руб\n"
            confirm += f"📉 Расходы: {td['expenses']:.2f} руб\n"
            confirm += f"🏦 Долги: {td['debt']:.2f} руб\n\n"
            confirm += "🔘 Нажмите «Подтвердить» для генерации отчёта."
            markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.row("✅ Подтвердить", "✏️ Заново")
            markup.row("◀️ Назад")
            bot.send_message(user_id, confirm, reply_markup=markup, parse_mode='Markdown')
            set_state(user_id, STATES["MANUAL_CONFIRM"])
        except:
            bot.send_message(user_id, "❌ Введите число (можно использовать пробелы или запятые).", reply_markup=back_button())
    elif state == STATES["MANUAL_CONFIRM"] and text == "✅ Подтвердить":
        td = user["temp_data"]
        days = td['days']
        revenue = td['revenue']
        expenses = td['expenses']
        debt = td['debt']
        daily_rev = revenue / days
        daily_exp = expenses / days
        start_date = datetime.now() - timedelta(days=days-1)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        df_manual = pd.DataFrame({'Дата': dates, 'Выручка': [daily_rev]*days, 'Расходы': [daily_exp]*days})
        user["last_df"] = df_manual
        bot.send_message(user_id, "⏳ Генерирую отчёт...")
        metrics = calculate_metrics(df_manual)
        analyzer = AIAnalyzer(df_manual)
        fig_gap, conclusion = analyzer.predict_cash_gap()
        if fig_gap:
            fig_gap.savefig('cash_gap.png')
            with open('cash_gap.png', 'rb') as photo:
                bot.send_photo(user_id, photo, caption=f"🔮 {conclusion}")
            os.remove('cash_gap.png')
        send_brief_summary(user_id, metrics)
        charts, charts_dir = create_charts(df_manual, metrics)
        pdf_bytes = generate_pdf_report(metrics, df_manual, charts)
        excel_bytes = generate_excel_report(df_manual, metrics, manual_mode=True)
        bot.send_document(user_id, excel_bytes, visible_file_name="отчёт_ProfitBot.xlsx")
        time.sleep(1)
        bot.send_document(user_id, pdf_bytes, visible_file_name="отчёт_ProfitBot.pdf")
        shutil.rmtree(charts_dir)
        if not is_admin(user_id):
            user["free_used"] = True
        set_state(user_id, STATES["MAIN_MENU"])
        bot.send_message(user_id, "✅ Отчёт готов. Вернуться в меню: /start", reply_markup=main_menu_keyboard())
    elif state == STATES["MANUAL_CONFIRM"] and text == "✏️ Заново":
        user["temp_data"] = {}
        set_state(user_id, STATES["MANUAL_DAYS"])
        bot.send_message(user_id, "📅 За сколько дней данные? (например: 30):", reply_markup=back_button())
    elif state == STATES["ADMIN_CONTACT"]:
        for aid in ADMIN_IDS:
            bot.send_message(aid, f"📬 *Сообщение от пользователя {user_id}:*\n\n{text}")
        bot.send_message(user_id, "✅ Ваше сообщение отправлено администратору. Ожидайте ответа.")
        set_state(user_id, STATES["MAIN_MENU"])
        bot.send_message(user_id, "🏠 Главное меню", reply_markup=main_menu_keyboard())
    elif state == STATES["PAYMENT_DETAILS"] and text == "📤 Отправить чек":
        if is_admin(user_id):
            bot.send_message(user_id, "👑 Вы админ – оплата не требуется.")
            return
        set_state(user_id, STATES["AWAITING_PAYMENT_PROOF"])
        user["awaiting_payment"] = True
        bot.send_message(user_id, "📸 Пришлите скриншот или фото чека.", reply_markup=back_button())

# ==================== ОБРАБОТЧИК ФАЙЛОВ ====================
@bot.message_handler(content_types=['document'])
def handle_excel(message):
    user_id = message.chat.id
    user = get_user(user_id)
    if user["state"] != STATES["AWAITING_EXCEL"]:
        bot.send_message(user_id, "❌ Сначала нажмите кнопку «📥 Загрузить Excel» в меню.")
        return
    can, msg = can_generate_report(user_id, user)
    if not can:
        bot.send_message(user_id, msg)
        set_state(user_id, STATES["MAIN_MENU"])
        return
    file_name = message.document.file_name
    if not (file_name.endswith('.xlsx') or file_name.endswith('.xls')):
        bot.send_message(user_id, "❌ Пожалуйста, загрузите файл в формате .xlsx или .xls")
        return
    try:
        bot.send_message(user_id, "⏳ Обрабатываю файл...")
        file_info = bot.get_file(message.document.file_id)
        downloaded = bot.download_file(file_info.file_path)
        df = pd.read_excel(BytesIO(downloaded), engine='openpyxl')
        rename_dict = {}
        for col in df.columns:
            col_low = str(col).strip().lower()
            if col_low in ['дата', 'date']:
                rename_dict[col] = 'Дата'
            elif col_low in ['выручка', 'revenue', 'доход']:
                rename_dict[col] = 'Выручка'
            elif col_low in ['расходы', 'expense', 'затраты']:
                rename_dict[col] = 'Расходы'
        if rename_dict:
            df = df.rename(columns=rename_dict)
        if 'Дата' not in df.columns or 'Выручка' not in df.columns or 'Расходы' not in df.columns:
            bot.send_message(user_id, "❌ Ошибка: нужны колонки 'Дата', 'Выручка', 'Расходы'")
            return
        df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
        df['Выручка'] = pd.to_numeric(df['Выручка'], errors='coerce')
        df['Расходы'] = pd.to_numeric(df['Расходы'], errors='coerce')
        df = df.dropna(subset=['Дата', 'Выручка', 'Расходы'])
        if df.empty:
            bot.send_message(user_id, "❌ Нет валидных данных.")
            return
        user["last_df"] = df
        bot.send_message(user_id, f"✅ Файл загружен! {len(df)} строк с данными.")
        metrics = calculate_metrics(df)
        analyzer = AIAnalyzer(df)
        fig_gap, conclusion = analyzer.predict_cash_gap()
        if fig_gap:
            fig_gap.savefig('cash_gap.png')
            with open('cash_gap.png', 'rb') as photo:
                bot.send_photo(user_id, photo, caption=f"🔮 {conclusion}")
            os.remove('cash_gap.png')
        send_brief_summary(user_id, metrics)
        charts, charts_dir = create_charts(df, metrics)
        pdf_bytes = generate_pdf_report(metrics, df, charts)
        excel_bytes = generate_excel_report(df, metrics)
        bot.send_document(user_id, excel_bytes, visible_file_name="отчёт_ProfitBot.xlsx")
        time.sleep(1)
        bot.send_document(user_id, pdf_bytes, visible_file_name="отчёт_ProfitBot.pdf")
        shutil.rmtree(charts_dir)
        if not is_admin(user_id):
            if not user["free_used"]:
                user["free_used"] = True
            else:
                user["paid"] = False
        set_state(user_id, STATES["MAIN_MENU"])
        bot.send_message(user_id, "✅ Отчёт готов. /start — меню", reply_markup=main_menu_keyboard())
    except Exception as e:
        bot.send_message(user_id, f"❌ Ошибка: {str(e)[:150]}")
        set_state(user_id, STATES["MAIN_MENU"])

# ==================== ФОТО (ЧЕКИ) ====================
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    user_id = message.chat.id
    user = get_user(user_id)
    if user["state"] == STATES["AWAITING_PAYMENT_PROOF"] and user.get("awaiting_payment"):
        if is_admin(user_id):
            bot.send_message(user_id, "👑 Вы админ – оплата не требуется.")
            return
        file_id = message.photo[-1].file_id
        for aid in ADMIN_IDS:
            bot.send_photo(aid, file_id, caption=f"🧾 *Чек от пользователя {user_id}*\n/approve {user_id}", parse_mode='Markdown')
        bot.send_message(user_id, "✅ Чек отправлен администратору.")
        set_state(user_id, STATES["MAIN_MENU"])
        user["awaiting_payment"] = True
    else:
        bot.send_message(user_id, "📸 Если это чек для оплаты, нажмите «💳 Детали оплаты» → «📤 Отправить чек».")

# ==================== АДМИН-КОМАНДЫ ====================
@bot.message_handler(commands=['approve'])
def approve_payment(message):
    if not is_admin(message.chat.id):
        return
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "❓ Использование: /approve user_id")
        return
    try:
        uid = int(parts[1])
        if uid in users_db and users_db[uid].get("awaiting_payment"):
            users_db[uid]["paid"] = True
            users_db[uid]["awaiting_payment"] = False
            bot.send_message(uid, "✅ Оплата подтверждена! Теперь вы можете генерировать отчёты.")
            bot.reply_to(message, f"✅ Пользователь {uid} получил доступ.")
        else:
            bot.reply_to(message, "❌ Пользователь не найден или не ожидает оплаты.")
    except:
        bot.reply_to(message, "❌ Ошибка: укажите числовой ID.")

@bot.message_handler(commands=['reply'])
def admin_reply(message):
    if not is_admin(message.chat.id):
        return
    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        return
    try:
        uid = int(parts[1])
        bot.send_message(uid, f"📨 *Ответ администратора:* {parts[2]}", parse_mode='Markdown')
    except:
        pass

@bot.message_handler(commands=['broadcast'])
def broadcast(message):
    if not is_admin(message.chat.id):
        return
    text = message.text.replace('/broadcast', '', 1).strip()
    if not text:
        return
    cnt = 0
    for uid in users_db:
        try:
            bot.send_message(uid, text)
            cnt += 1
        except:
            pass
    bot.reply_to(message, f"📢 Рассылка отправлена {cnt} пользователям.")

@bot.message_handler(commands=['stats'])
def stats(message):
    if not is_admin(message.chat.id):
        return
    total = len(users_db)
    paid = sum(1 for u in users_db.values() if u.get("paid") or u.get("free_used"))
    bot.reply_to(message, f"📊 *Пользователей:* {total}\n📄 *Получили отчёт:* {paid}", parse_mode='Markdown')

# ==================== FLASK МАРШРУТЫ ====================
@flask_app.route('/', methods=['GET', 'HEAD'])
def health():
    return "ProfitBot is running!", 200

@flask_app.route('/health')
def health_check():
    return "OK", 200

@flask_app.route('/webhook', methods=['POST'])
def webhook():
    update = telebot.types.Update.de_json(request.stream.read().decode('utf-8'))
    bot.process_new_updates([update])
    return 'ok', 200

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    # Устанавливаем webhook вместо polling
    webhook_url = f"https://{os.environ.get('RENDER_EXTERNAL_URL', 'profitbot1.onrender.com')}/webhook"
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=webhook_url)
    print(f"✅ Webhook установлен: {webhook_url}")
    
    # Запускаем Flask
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)
