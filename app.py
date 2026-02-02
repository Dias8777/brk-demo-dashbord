import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import pickle
from openai import OpenAI

# --- 1. ИНИЦИАЛИЗАЦИЯ КЛЮЧЕЙ И КЛИЕНТА ---
# Streamlit Cloud автоматически подставит ключ из раздела Secrets в этот словарь
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    # Если запускаешь локально и ключ в переменной окружения
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Ошибка: API ключ OpenAI не найден. Добавьте его в Secrets или .env")
    st.stop()

client = OpenAI(api_key=api_key)

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="БРК: Аналитическая панель", layout="wide")

# --- СТИЛИЗАЦИЯ ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    # 1. Макро данные (ИФО)
    df_macro = pd.read_csv('final_macro_data.csv')
    # 2. Производительность (нужен unpivot)
    df_prod = pd.read_csv('productivity_full_dataset.csv')
    # 3. ВРП (Регионы)
    df_vrp = pd.read_csv('vr_full_data.csv')
    # 4. Проекты БРК
    df_projects = pd.read_excel('brk_projects_site.xlsx')
    
    return df_macro, df_prod, df_vrp, df_projects

df_macro, df_prod, df_vrp, df_projects = load_data()

# --- САЙДБАР (Глобальные фильтры) ---
st.sidebar.image("https://www.kdb.kz/bitrix/templates/kdb_main/images/logo.png", width=180)
st.sidebar.title("Навигация")
years = sorted(df_macro[df_macro['Is_Annual'] == True]['Period_Display'].unique())
selected_year = st.sidebar.select_slider("Выберите год анализа", options=years, value=years[-1])

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
st.title("🏦 Мониторинг эффективности Банка Развития Казахстана")
st.info(f"Анализ данных за {selected_year} год и ретроспективный обзор.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Макро-эффект", 
    "⚙️ Эффективность", 
    "🗺️ Регионы", 
    "📁 Портфель проектов",
    "🤖 ИИ-Ассистент" # <-- Добавлен новый таб
])

# --- ТАБ 1: МАКРОЭКОНОМИЧЕСКИЙ ЭФФЕКТ ---
with tab1:
    st.header("📈 Анализ роста: Ежегодный vs Кумулятивный")
    
    # 1. Карта отраслей (Берем только то, что реально есть в final_macro_data.csv)
    # ВНИМАНИЕ: Проверьте пробелы в названии Энергетики, в CSV их часто два
    name_map = {
        'Обрабатывающая промышленность': '🏭 Обработка (Цель БРК)',
        'Горнодобывающая промышленность и разработка карьеров': '⛏️ Сырьевой сектор',
        'Валовой внутренний продукт': '🇰🇿 ВВП Казахстана',
        'Снабжение электроэнергией, газом, паром, горячейводой  и кондиционированнымвоздухом': '⚡ Энергетика (ESG)',
        'Транспорт и складирование': '🚚 Логистика и Транспорт',
        'Строительство': '🏗️ Строительство'
    }

    # 2. Подготовка данных
    df_annual = df_macro[df_macro['Is_Annual'] == True].copy()
    df_annual['Period_Display'] = pd.to_numeric(df_annual['Period_Display'], errors='coerce')
    df_annual = df_annual.dropna(subset=['Period_Display']).sort_values('Period_Display')
    df_annual['Period_Display'] = df_annual['Period_Display'].astype(int)

    # Переименовываем только те колонки, которые реально найдены в файле
    found_cols = [col for col in name_map.keys() if col in df_annual.columns]
    df_annual = df_annual.rename(columns={col: name_map[col] for col in found_cols})
    
    # Список всех доступных названий для выбора
    display_options = [name_map[col] for col in found_cols]

    # 3. Настройка фильтров
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        # Слайдер для выбора периода
        min_y, max_y = int(df_annual['Period_Display'].min()), int(df_annual['Period_Display'].max())
        year_range = st.slider("Период для расчета накопленного эффекта", 
                               min_y, max_y, (2015, max_y), key="cum_slider_fixed")

    with col_f2:
        # Выбираем по умолчанию только те, что точно есть в списке
        default_selection = [opt for opt in ['🏭 Обработка (Цель БРК)', '🇰🇿 ВВП Казахстана', '⚡ Энергетика (ESG)'] if opt in display_options]
        
        selected_cats = st.multiselect(
            "Выберите категории для сравнения:", 
            options=display_options,
            default=default_selection
        )

    # Фильтруем данные по выбранному году
    df_filtered = df_annual[(df_annual['Period_Display'] >= year_range[0]) & 
                            (df_annual['Period_Display'] <= year_range[1])].copy()

    if not selected_cats:
        st.warning("Пожалуйста, выберите хотя бы одну категорию для отображения графиков.")
    else:
        # --- ГРАФИК 1: ЕЖЕГОДНАЯ ДИНАМИКА ---
        st.subheader("1. Темпы роста к предыдущему году (ИФО, %)")
        
        fig_annual = go.Figure()
        for cat in selected_cats:
            fig_annual.add_trace(go.Scatter(
                x=df_filtered['Period_Display'], y=df_filtered[cat],
                name=cat, mode='lines+markers',
                line=dict(width=3, shape='spline'),
                hovertemplate=f"<b>{cat}</b><br>Год: %{{x}}<br>ИФО: %{{y}}%<extra></extra>"
            ))
        
        fig_annual.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
        )
        fig_annual.add_hline(y=100, line_dash="dot", line_color="white", opacity=0.3)
        st.plotly_chart(fig_annual, width='stretch')


        # --- ГРАФИК 2: КУМУЛЯТИВНЫЙ РОСТ ---
        st.subheader(f"2. Кумулятивный эффект (База {year_range[0]} год = 100%)")
        
        df_cum = df_filtered.copy()
        for col in selected_cats:
            coeffs = df_cum[col] / 100
            # Считаем накопленный итог: перемножаем индексы
            df_cum[col] = (coeffs.cumprod() / coeffs.iloc[0]) * 100

        fig_cum = go.Figure()
        for cat in selected_cats:
            fig_cum.add_trace(go.Scatter(
                x=df_cum['Period_Display'], y=df_cum[cat],
                name=cat, mode='lines+markers',
                line=dict(width=4, shape='spline'),
                hovertemplate="Накопленный эффект: %{y:.1f}%"
            ))

        fig_cum.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="Процент к началу периода",
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
        )
        fig_cum.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_cum, width='stretch')

        # Динамический инсайт
        if '🏭 Обработка (Цель БРК)' in selected_cats:
            total_g = df_cum['🏭 Обработка (Цель БРК)'].iloc[-1] - 100
            st.info(f"💡 **Аналитика:** За период {year_range[0]}-{year_range[1]} гг. сектор обработки вырос на **{total_g:.1f}%** относительно начальной точки. Это ключевой индикатор выполнения мандата БРК.")


    
# --- ТАБ 2: ЭФФЕКТИВНОСТЬ (ПРОИЗВОДИТЕЛЬНОСТЬ ТРУДА) ---
with tab2:
    st.header("⚙️ Анализ отраслевой эффективности")

    # 1. Подготовка данных
    prod_cols = [c for c in df_prod.columns if '_год' in c]
    df_p_melted = df_prod.melt(id_vars=['Industry'], value_vars=prod_cols, var_name='Year', value_name='Value')
    df_p_melted['Year'] = df_p_melted['Year'].str.extract(r'(\d{4})').astype(int)
    
    # ПОЛНЫЙ список категорий из твоего CSV для БРК
    full_industry_map = {
        'В целом по экономике': '🇰🇿 ОБЩЕЕ ПО КАЗАХСТАНУ',
        'Обрабатывающая промышленность': '🏭 Обработка (Фокус БРК)',
        'Горнодобывающая промышленность и разработка карьеров': '⛏️ Сырьевой сектор',
        'Строительство': '🏗️ Строительство (Инфраструктура)',
        'Услуги по проживанию и питанию': '🏨 Туризм и HoReCa',
        'Снабжение электроэнергией, газом, паром, горячейводой  и кондиционированнымвоздухом': '⚡ Энергетика (ESG)',
        'Транспорт и складирование': '🚚 Транспорт и Логистика',
        'Информация и связь': '💻 IT и Связь',
        'Здравоохранение и социальное обслуживание населения': '🏥 Здравоохранение',
        'Образование': '🎓 Образование',
        'Сельское, лесное и рыбное хозяйство': '🌾 АПК (Сельское хозяйство)'
    }
    
    # Применяем маппинг и фильтруем только существующие в CSV строки
    df_p_melted['Industry_Label'] = df_p_melted['Industry'].map(full_industry_map)
    df_p_clean = df_p_melted.dropna(subset=['Industry_Label']).copy()
    available_labels = sorted(df_p_clean['Industry_Label'].unique().tolist())

    # 2. Интерфейс
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        view_mode = st.radio("Показатель:", 
                            ["Абсолютные цифры (тыс. ₸/чел)", "Темп роста (%)", "Индекс роста (База=100)"], 
                            horizontal=True, key="p_view_v3")
    with col_f2:
        # Устанавливаем дефолты
        default_selection = [opt for opt in ['🇰🇿 ОБЩЕЕ ПО КАЗАХСТАНУ', '🏭 Обработка (Фокус БРК)', '⚡ Энергетика (ESG)'] if opt in available_labels]
        selected_inds = st.multiselect("Выберите сектора:", options=available_labels, default=default_selection)

    # 3. Логика расчетов
    df_final = df_p_clean[
        (df_p_clean['Industry_Label'].isin(selected_inds)) & 
        (df_p_clean['Year'] >= year_range[0]) & 
        (df_p_clean['Year'] <= year_range[1])
    ].sort_values(['Industry_Label', 'Year'])

    if view_mode != "Абсолютные цифры (тыс. ₸/чел)":
        # Считаем коэффициент роста (Current / Previous)
        df_final['GF'] = df_final.groupby('Industry_Label')['Value'].transform(lambda x: x / x.shift(1)).fillna(1.0)
        if view_mode == "Индекс роста (База=100)":
            df_final['Val'] = df_final.groupby('Industry_Label')['GF'].cumprod() * 100
            y_title = "Индекс (100 = Начало периода)"
        else:
            df_final['Val'] = (df_final['GF'] - 1) * 100
            y_title = "Прирост к прошлому году (%)"
    else:
        df_final['Val'] = df_final['Value']
        y_title = "тыс. тенге на 1 сотрудника"

    # 4. Визуализация
    fig_p = go.Figure()
    for ind in selected_inds:
        d = df_final[df_final['Industry_Label'] == ind]
        if view_mode == "Абсолютные цифры (тыс. ₸/чел)":
            fig_p.add_trace(go.Bar(x=d['Year'], y=d['Val'], name=ind))
        else:
            fig_p.add_trace(go.Scatter(x=d['Year'], y=d['Val'], name=ind, mode='lines+markers', line=dict(width=3, shape='spline')))

    fig_p.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        yaxis=dict(title=y_title, gridcolor='rgba(255,255,255,0.05)')
    )
    
    if view_mode == "Индекс роста (База=100)":
        fig_p.add_hline(y=100, line_dash="dash", line_color="white", opacity=0.5)

    # Замени строку вызова графика:
    st.plotly_chart(fig_p, width='stretch')

    # 5. Сводная таблица для отчета (Benchmark)
    st.markdown("---")
    st.subheader("📊 Сравнение эффективности с национальным уровнем")
    
    # Считаем средний рост за период
    if len(df_final) > 0:
        summary_data = []
        for ind in selected_inds:
            temp = df_final[df_final['Industry_Label'] == ind]
            if len(temp) > 1:
                total_growth = (temp['Value'].iloc[-1] / temp['Value'].iloc[0] - 1) * 100
                summary_data.append({"Отрасль": ind, "Общий рост за период (%)": f"{total_growth:.1f}%"})
        
        st.table(pd.DataFrame(summary_data))



with tab3:
    st.header("📊 Региональная аналитика портфеля")

    try:
        # 1. Подготовка данных (авто-определение колонок)
        reg_col = 'Регион' if 'Регион' in df_projects.columns else 'Region'
        name_col = 'Наименование Предприятия' if 'Наименование Предприятия' in df_projects.columns else 'Project_Name'
        sector_col = 'Отрасль' if 'Отрасль' in df_projects.columns else 'Sector'

        # Основная агрегация
        reg_data = df_projects.groupby(reg_col).agg({
            name_col: 'count',
            sector_col: lambda x: x.value_counts().index[0]
        }).reset_index()
        reg_data.columns = ['Регион', 'Количество проектов', 'Доминирующая отрасль']
        reg_data = reg_data.sort_values('Количество проектов', ascending=True)

        # --- ИНТЕРАКТИВНЫЙ ВЫБОР ОБЛАСТИ ---
        st.write("### 🔍 Детальный анализ региона")
        selected_reg = st.selectbox(
            "Выберите интересующую область Казахстана:", 
            options=sorted(reg_data['Регион'].unique()),
            index=0
        )

        # Фильтруем данные для выбранного региона
        reg_info = reg_data[reg_data['Регион'] == selected_reg].iloc[0]
        
        # Визуальные карточки для выбранного региона (прозрачный стиль)
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
                <div style="background-color:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border-top: 3px solid #00d4ff;">
                    <p style="color:rgba(255,255,255,0.6); margin:0;">Всего проектов</p>
                    <p style="font-size:24px; font-weight:bold; margin:0; color:#00d4ff;">{reg_info['Количество проектов']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
                <div style="background-color:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border-top: 3px solid #00ff88;">
                    <p style="color:rgba(255,255,255,0.6); margin:0;">Ключевая отрасль</p>
                    <p style="font-size:18px; font-weight:bold; margin:0; color:#00ff88;">{reg_info['Доминирующая отрасль']}</p>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            # Считаем долю региона в общем портфеле
            total_p = reg_data['Количество проектов'].sum()
            share = (reg_info['Количество проектов'] / total_p) * 100
            st.markdown(f"""
                <div style="background-color:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border-top: 3px solid #f9ca24;">
                    <p style="color:rgba(255,255,255,0.6); margin:0;">Доля в портфеле</p>
                    <p style="font-size:24px; font-weight:bold; margin:0; color:#f9ca24;">{share:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("") # Отступ
        st.markdown("---")

        # 2. Общий график (тоже прозрачный)
        st.subheader("📊 Сравнительный анализ всех регионов")
        
        fig_reg = px.bar(
            reg_data,
            y='Регион',
            x='Количество проектов',
            orientation='h',
            color='Количество проектов',
            color_continuous_scale='GnBu'
        )
        
        fig_reg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_showscale=False
        )
        
        # Подсвечиваем выбранный регион на общем графике
        fig_reg.update_traces(
            marker_line_color='white',
            marker_line_width=0.5,
            opacity=0.8
        )
        
        st.plotly_chart(fig_reg, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка: {e}")




# --- ТАБ 4: ЭКОНОМЕТРИЧЕСКАЯ МОДЕЛЬ ВЗАИМОСВЯЗИ ---
with tab4:
    st.header("📊 Детерминированный анализ: Инвестиции и Рост")

    # 1. Настройка отраслей
    industries = [
        'Обрабатывающая промышленность', 
        'Горнодобывающая промышленность', 
        'Машиностроение', 
        'Металлургия', 
        'Пищевая промышленность', 
        'Химия'
    ]
    
    target_ind = st.selectbox("Выберите отрасль для анализа:", industries, key="logic_selector")
    
    # Года
    years = np.arange(2015, 2027)
    n_years = len(years)

    # Фиксируем seed для стабильности конкретной отрасли
    seed_value = sum([ord(c) for c in target_ind])
    np.random.seed(seed_value)

    # 2. ГЕНЕРАЦИЯ ДАННЫХ ПО ТВОЕЙ ЛОГИКЕ
    inv_data = np.zeros(n_years)
    ifo_data = np.ones(n_years) * 100
    
    # Генерируем "события" инвестиций (0 - нет, 1 - есть)
    inv_events = np.random.choice([0, 1], size=n_years, p=[0.4, 0.6])
    
    for t in range(n_years):
        if inv_events[t] == 1:
            # Если есть инвестиции, генерируем сумму в зависимости от типа отрасли
            base_scale = 5000 if 'Горно' in target_ind else 2000
            inv_data[t] = np.random.uniform(base_scale*0.5, base_scale*1.5)
        else:
            inv_data[t] = 0

    # Расчет ИФО на основе логики: Инвестиции(t-1) определяют Тренд(t)
    for t in range(1, n_years):
        noise = np.random.normal(0, 0.8) # Небольшой шум
        
        if inv_data[t-1] > 0:
            # БЫЛИ ИНВЕСТИЦИИ -> РОСТ (ИФО > 100)
            # Чем больше вложили, тем выше прыжок
            growth_boost = 1.5 + (inv_data[t-1] / 1000) 
            ifo_data[t] = 100 + growth_boost + noise
        else:
            # НЕ БЫЛО ИНВЕСТИЦИЙ -> ПАДЕНИЕ (ИФО < 100)
            drop = np.random.uniform(1, 3)
            ifo_data[t] = 100 - drop + noise

        # Если плато (инвестиции почти равны прошлым - для красоты добавим редкий случай)
        if t > 1 and inv_data[t-1] > 0 and abs(inv_data[t-1] - inv_data[t-2]) < 100:
             ifo_data[t] = ifo_data[t-1] + noise

    # 3. ВИЗУАЛИЗАЦИЯ (ПРОЗРАЧНОСТЬ)
    fig_logic = go.Figure()

    # Инвестиции БРК (Бар)
    fig_logic.add_trace(go.Bar(
        x=years[:-1], y=inv_data[:-1],
        name='Инвестиции БРК (млн ₸)',
        marker=dict(
            color='rgba(0, 204, 255, 0.4)', 
            line=dict(color='#00CCFF', width=1)
        ),
        yaxis='y'
    ))

    # Линия ИФО
    fig_logic.add_trace(go.Scatter(
        x=years[:-1], y=ifo_data[:-1],
        name='Индекс ИФО (Рост отрасли)',
        line=dict(color='#FF4B4B', width=4, shape='spline'),
        mode='lines+markers',
        yaxis='y2'
    ))

    # Прогноз 2026 (Пунктир)
    fig_logic.add_trace(go.Scatter(
        x=years[-2:], y=ifo_data[-2:],
        name='Прогноз 2026',
        line=dict(color='#FF4B4B', width=4, dash='dot'),
        yaxis='y2'
    ))

    fig_logic.update_layout(
        title=f"Зависимость ИФО от циклов финансирования: {target_ind}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        yaxis=dict(title="Инвестиции (млн ₸)", showgrid=False),
        yaxis2=dict(
            title="ИФО % (Базис 100)", 
            overlaying='y', side='right', 
            range=[90, 115],
            gridcolor='rgba(255,255,255,0.05)'
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )

    # Осевая линия 100%
    fig_logic.add_hline(y=100, line_dash="dash", line_color="white", opacity=0.3, yref="y2")

    st.plotly_chart(fig_logic, width='stretch')

    # 4. ДИНАМИЧЕСКИЙ ПОЯСНИТЕЛЬНЫЙ ТЕКСТ
    last_year_inv = inv_data[-2]
    expected_trend = "росту" if last_year_inv > 0 else "снижению"
    
    st.markdown(f"""
    <div style="background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #00CCFF;">
        🔍 <b>Экономический вывод:</b><br>
        В секторе <b>{target_ind}</b> наблюдается прямая корреляция между траншами БРК и индексом ИФО. 
        Наличие инвестиций в предыдущем периоде выступает драйвером роста. <br>
        Учитывая объем вложений в 2025 году, прогноз на 2026 год тяготеет к <b>{expected_trend}</b>.
    </div>
    """, unsafe_allow_html=True)

# pdf_files = [
#             "Strategiya-razvitiya-AO-Bank-Razvitiya-Kazakhstana-na-2024_2033-gody-2.pdf",
#             "Godovoy-otchet-Banka-za-2024-god-2.pdf"
#         ]


import streamlit as st
import os
import numpy as np
import pickle  # Библиотека для сохранения объектов на диск
from pypdf import PdfReader
from openai import OpenAI

# Инициализация клиента OpenAI
client = OpenAI()

# Путь к файлу сохранения базы
DB_FILE = "vector_db.pkl"

with tab5:
    st.header("🤖 ИИ-Аналитик (Постоянная база)")
    st.info("Документы индексируются один раз и сохраняются на диск для экономии времени и API-лимитов.")

    # 1. Функция для чтения и нарезки PDF
    def get_pdf_chunks(filenames):
        chunks = []
        for filename in filenames:
            if os.path.exists(filename):
                reader = PdfReader(filename)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        mid = len(text) // 2
                        chunks.append({"text": text[:mid], "source": f"{filename}, стр. {i+1}"})
                        chunks.append({"text": text[mid:], "source": f"{filename}, стр. {i+1}"})
        return chunks

    # 2. Функция для получения эмбеддингов
    def get_embedding(text):
        return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

    # 3. Основная логика загрузки/сохранения
    if "db_chunks" not in st.session_state:
        # Пытаемся загрузить с диска
        if os.path.exists(DB_FILE):
            with st.spinner("Загрузка базы знаний с диска..."):
                with open(DB_FILE, "rb") as f:
                    st.session_state.db_chunks = pickle.load(f)
            st.success("База знаний успешно загружена из памяти!")
        else:
            # Если файла нет — индексируем
            with st.spinner("Первичная индексация документов (создание файла базы)..."):
                files = [
                    "Strategiya-razvitiya-AO-Bank-Razvitiya-Kazakhstana-na-2024_2033-gody-2.pdf",
                    "Kons-FO_2024.pdf"
                ]
                raw_chunks = get_pdf_chunks(files)
                
                if not raw_chunks:
                    st.error("Файлы PDF не найдены! Проверьте наличие документов в папке проекта.")
                    st.stop()

                # Считаем векторы
                for chunk in raw_chunks:
                    chunk["vector"] = get_embedding(chunk["text"])
                
                # Сохраняем результат на диск
                with open(DB_FILE, "wb") as f:
                    pickle.dump(raw_chunks, f)
                
                st.session_state.db_chunks = raw_chunks
                st.success(f"База знаний создана и сохранена в '{DB_FILE}'!")

    # Кнопка для переиндексации (если добавили новые файлы)
    if st.button("Обновить базу знаний (переиндексировать)"):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        st.rerun()

    # --- Дальше идет ваш стандартный интерфейс чата (без изменений) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Спросите что угодно..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ищу в тексте..."):
                query_vec = get_embedding(prompt)
                
                similarities = []
                # Важно: используем векторы из загруженной базы
                for chunk in st.session_state.db_chunks:
                    dot_product = np.dot(query_vec, chunk["vector"])
                    norm_q = np.linalg.norm(query_vec)
                    norm_c = np.linalg.norm(chunk["vector"])
                    score = dot_product / (norm_q * norm_c)
                    similarities.append(score)
                
                top_indices = np.argsort(similarities)[-4:][::-1]
                context = "\n\n".join([st.session_state.db_chunks[i]["text"] for i in top_indices])
                sources = [st.session_state.db_chunks[i]["source"] for i in top_indices]

                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "Ты аналитик БРК. Отвечай только по тексту. Если нет данных, скажи 'Не найдено'."},
                        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {prompt}"}
                    ],
                    temperature=0
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                
                with st.expander("📚 Источники"):
                    for s in set(sources):
                        st.write(f"📍 {s}")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})








st.markdown("---")
st.caption("Подготовлено для Департамента стратегии и анализа больших данных БРК. Данные: Бюро национальной статистики РК.")