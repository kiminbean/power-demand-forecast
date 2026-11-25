def create_gauge_chart(value: float, title: str, max_val: float = 100,
                       color: str = '#3B82F6', suffix: str = '',
                       reserve: bool = False) -> go.Figure:
    """게이지. 예비율은 낮을수록 위험한 구간색을 쓴다.

    T7(test_v4_reserve_gauge)이 이 함수를 go만 있는 이름공간에서 실행하므로
    리터럴과 builtins만 쓴다. 글자색 '#000037'은 Streamlit 1.52.1 Plotly 테마의
    자리표시자(getGray90: 라이트 #262730 / 다크 #FAFAFA)라 두 테마에서 4.5:1을 넘는다.
    """
    if reserve:
        # 예비율은 낮을수록 위험하며 표시 범위는 실제 값을 포함한다.
        low = min(0, value)
        max_val = max(max_val, value * 1.1)
        steps = [
            {'range': [low, 10], 'color': 'rgba(220, 38, 38, 0.22)'},
            {'range': [10, 15], 'color': 'rgba(180, 83, 9, 0.22)'},
            {'range': [15, max_val], 'color': 'rgba(22, 163, 74, 0.18)'},
        ]
        threshold = 10
    else:
        low = 0
        steps = [
            {'range': [0, max_val * 0.5], 'color': 'rgba(100, 116, 139, 0.10)'},
            {'range': [max_val * 0.5, max_val * 0.8], 'color': 'rgba(100, 116, 139, 0.18)'},
        ]
        threshold = max_val * 0.9
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix, 'valueformat': '.1f' if reserve else ',.0f',
                'font': {'size': 28, 'color': '#000037'}},
        title={'text': title, 'font': {'size': 13, 'color': '#000037'}},
        gauge={
            'axis': {'range': [low, max_val], 'nticks': 6,
                     'tickfont': {'size': 12, 'color': '#000037'}},
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': 'rgba(100, 116, 139, 0.10)',
            'borderwidth': 0,
            'steps': steps,
            'threshold': {'line': {'color': '#DC2626', 'width': 2.5}, 'thickness': 0.85, 'value': threshold},
        },
    ))
    fig.update_layout(margin=dict(l=28, r=28, t=16, b=8), height=190, dragmode=False)
    return fig


def main():
    """메인 함수 (빌드 명세 2026-11-23 골격: 테스트 제약 문장은 그대로, 나머지는 st.html)"""
    st.html(_css_dashboard())  # <style>만 담아 레이아웃 공간 0

    # 데이터 로드 (기존과 같음)
    plants_df = get_jeju_power_plants()
    power_status = get_current_power_status()
    smp_data = get_smp_data()
    backend_predictions = get_backend_smp_predictions()
    if backend_predictions:
        smp_data = {**smp_data, **backend_predictions}
    weather = get_weather_data()
    api_kpis = fetch_backend_json("/api/v1/dashboard/kpis")  # 캐시 재사용: 출처 라벨 전용
    kpx_local = fetch_jeju_realtime()                        # 캐시 재사용: 공급능력 출처 라벨 전용
    registry = load_power_plants_data()                       # 캐시 재사용: 지도 밖 자원·등록부 기준일

    admin_mode = os.environ.get("POWERDEMAND_DASHBOARD_ADMIN", "").strip() == "1"
    test_alert = False
    test_reserve = None

    with st.sidebar:
        st.html(_html_sidebar_section("보기"))
        auto_refresh = st.toggle("60초마다 자동 갱신", value=True, key="auto_refresh")
        st.html(_html_theme_hint())
        st.html(_html_sidebar_section("경보 이력 · 최근 7일"))
        alert_history = get_alert_history()
        st.html(_html_alert_history(_fmt_alert_history(alert_history.get_recent(100), datetime.now())))
        if admin_mode:
            st.html(_html_sidebar_section("관리자 도구"))
            test_alert = st.toggle("알림 테스트 모드", value=False, key="admin_test_alert")
            if test_alert:
                test_reserve = st.slider("테스트 예비율 (%)", min_value=0.0, max_value=30.0, value=12.0, step=1.0)
                power_status = dict(power_status)
                power_status['reserve_rate'] = test_reserve
                power_status['data_source'] = '테스트 모드'
            if st.button("경보 이력 초기화", key="clear_history", icon=":material/delete:"):
                _render_clear_history_dialog(alert_history)  # @st.dialog 확인 뒤 clear()

    # 경보 판정·기록·발송: 전체 실행 경로에만 둔다 (자동 갱신 fragment 안 금지)
    reserve_rate = power_status['reserve_rate']
    level = _fmt_reserve_level(reserve_rate)
    if level['show_alert'] and not test_alert:
        alert_history = get_alert_history()
        alert_history.add_alert(reserve_rate=reserve_rate, status=level['status'],
                                title=level['alert_title'], message=level['alert_msg'])
        if level['status'] == "critical":
            email_notifier = get_email_notifier()
            if email_notifier.is_configured():
                success, email_msg = email_notifier.send_critical_alert(
                    reserve_rate=reserve_rate, status=level['status'], title=level['alert_title'],
                    message=level['alert_msg'], power_data=power_status)
                if success:
                    st.toast("이메일 발송 완료", icon=":material/mail:")
        slack_notifier = get_slack_notifier()
        if slack_notifier.is_configured():
            success, slack_msg = slack_notifier.send_alert(
                reserve_rate=reserve_rate, status=level['status'], title=level['alert_title'],
                message=level['alert_msg'], power_data=power_status)
            if success:
                st.toast("Slack 알림 발송 완료", icon=":material/chat:")

    run_every = "60s" if auto_refresh else None

    @st.fragment(run_every=run_every)
    def live_header():
        power_status = dict(get_current_power_status())
        if test_alert:
            power_status['reserve_rate'] = test_reserve
            power_status['data_source'] = '테스트 모드'
        smp_data = get_smp_data()
        backend_predictions = get_backend_smp_predictions()
        if backend_predictions:
            smp_data = {**smp_data, **backend_predictions}
        weather = dict(get_weather_data())
        api_kpis = fetch_backend_json("/api/v1/dashboard/kpis")
        prov = _fmt_provenance(power_status, api_kpis, fetch_jeju_realtime())
        power_status.update(prov['display_keys'])      # pd_chip, pd_src_label, pd_state_class
        weather['pd_src'] = _fmt_weather_provenance(api_kpis)
        smp_data = {**smp_data, **_fmt_smp_display_keys(smp_data, datetime.now())}
        reserve_rate = power_status['reserve_rate']
        level = _fmt_reserve_level(reserve_rate)
        # 예비율이 실측이 아니면(대체값·기본값) 단계 이름 뒤에 '실측 아님'을 붙이고 빗금 클래스를 더한다
        reserve_class = level['class'] + ('' if prov['demand_measured'] else ' is-est')
        reserve_text = level['text'] + ('' if prov['demand_measured'] else ' · 실측 아님')
        demand_observed_text, demand_elapsed_text = _format_observation_age(
            power_status.get('data_date', 'N/A')
        )
        demand_observed_text, demand_elapsed_text = _fmt_observation_display(
            prov, demand_observed_text, demand_elapsed_text
        )

        st.html(_html_app_bar(power_status, smp_data, level, prov, auto_refresh))
        if level['show_alert']:
            st.html(_html_alert_banner(level, reserve_rate, prov))
        st.html(_html_status_line(power_status, level, prov))

        with st.container(key="kpi_row"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""<div class="pd-kpi {power_status.get('pd_state_class', '')}">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--bolt" aria-hidden="true"></span>현재 수요</span>{power_status.get('pd_chip', '')}</div>
<div class="pd-kpi__value"><span class="pd-num">{power_status['demand']:,.0f}</span><span class="pd-kpi__unit">MW</span></div>
<div class="pd-kpi__sub"><span class="pd-pill status-badge {reserve_class}"><span class="pd-pill__dot" aria-hidden="true"></span>예비율 {reserve_rate:.1f}% · {reserve_text}</span></div>
<div class="pd-kpi__meta"><span>출처 {power_status.get('pd_src_label', 'KPX 실시간')}</span><span>관측 {demand_observed_text} · {demand_elapsed_text}</span></div>
</div>""", unsafe_allow_html=True)
            with col2:
                # 이전 시간 실측값이 없으면 화살표/변화율 대신 안내 문구를 표시한다
                if smp_data.get('prev_available', False) and smp_data['change'] is not None:
                    smp_change_class = "is-up" if smp_data['change'] >= 0 else "is-down"
                    smp_arrow = "▲" if smp_data['change'] >= 0 else "▼"
                    smp_change_html = (
                        f'{smp_arrow} {abs(smp_data["change"]):.1f} 원/kWh ({smp_data["change_pct"]:+.1f}%) · 전 거래시간 대비'
                    )
                else:
                    smp_change_class = ""
                    smp_change_html = "이전 시간 데이터 없음"
                st.markdown(f"""<div class="pd-kpi">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--tag" aria-hidden="true"></span>제주 하루전 SMP</span>{'<span class="pd-chip pd-chip--measured"><span class="pd-chip__glyph" aria-hidden="true"></span>확정 가격</span>' if str(smp_data.get('data_source', '')).startswith('EPSIS') else '<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>시뮬레이션 · 실측 아님</span>'}{smp_data.get('pd_stale_chip', '')}</div>
<div class="pd-kpi__value"><span class="pd-num">{smp_data['current']:,.1f}</span><span class="pd-kpi__unit">원/kWh</span></div>
<div class="pd-kpi__sub"><span class="pd-delta {smp_change_class}">{smp_change_html}</span></div>
<div class="pd-kpi__meta"><span>출처 {smp_data.get('pd_src_label', 'EPSIS 하루전')}</span><span>거래시간 {smp_data.get('trade_label') or '확인 불가'}</span></div>
</div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="pd-kpi is-est">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--leaf" aria-hidden="true"></span>재생에너지 비율</span><span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>추정 · 실측 아님</span></div>
<div class="pd-kpi__value"><span class="pd-num">{power_status['renewable_ratio']:.1f}</span><span class="pd-kpi__unit">%</span></div>
<div class="pd-kpi__sub">태양광·풍력이 지금 수요를 채우는 비율</div>
<div class="pd-kpi__meta"><span>{power_status.get('renewable_source', '태양광 + 풍력 (추정치)')}</span><span>발전 현황 탭 구성비와 산출 근거 다름</span></div>
</div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class="pd-kpi{'' if weather.get('source') == 'kma' and weather.get('pd_src') != 'file' else ' is-est'}">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--thermo" aria-hidden="true"></span>기상 현황</span>{'<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>추정 · 과거 자료 평균</span>' if weather.get('pd_src') == 'file' else ('<span class="pd-chip pd-chip--measured"><span class="pd-chip__glyph" aria-hidden="true"></span>실측</span>' if weather.get('source') == 'kma' else '<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>시뮬레이션 · 실측 아님</span>')}</div>
<div class="pd-kpi__value"><span class="pd-num">{weather['temperature']:.1f}</span><span class="pd-kpi__unit">°C</span></div>
<div class="pd-kpi__sub">풍속 {weather['wind_speed']:.1f} m/s{(' · ' + str(weather.get('condition'))) if weather.get('condition') else ''}</div>
<div class="pd-kpi__meta"><span>{'기상청 과거 자료 24시간 평균 (API 서버) · 실시간 아님' if weather.get('pd_src') == 'file' else ('기상청 초단기실황 (API 서버)' if weather.get('source') == 'kma' else '시뮬레이션 값 (실측 아님)')}</span></div>
</div>""", unsafe_allow_html=True)

    live_header()
    st.html(_html_legend_strip())

    tab1, tab2, tab3, tab4 = st.tabs([":material/map: 실시간 현황", ":material/sell: SMP 예측",
                                      ":material/bolt: 발전 현황", ":material/insights: 분석"])

    with tab1:
        with st.container(border=True, key="map_stage"):
            with st.container(key="map_head"):
                head_l, head_r = st.columns([3, 1], vertical_alignment="center")
                with head_l:
                    st.html(_html_map_head(plants_df, registry))
                with head_r:
                    show_heatmap = st.toggle("추정 출력 열지도", value=False, key="map_heatmap")
            with st.container(key="map_body"):
                map_col, insp_col = st.columns([8, 4], gap="medium")
                with map_col:
                    with st.container(key="map_col"):
                        jeju_map = create_jeju_map(plants_df, show_heatmap)
                        st_folium(jeju_map, height=480, use_container_width=True, returned_objects=[], key="jeju_map")
                        st.html(_html_map_legend(plants_df))
                with insp_col:
                    map_region = st.segmented_control("지역", ["전체", "제주시", "서귀포시"], default="전체",
                                                      key="map_region", label_visibility="collapsed") or "전체"
                    st.html(_html_inspector(plants_df, registry, map_region))

        @st.fragment(run_every=run_every)
        def live_supply():
            power_status = dict(get_current_power_status())
            if test_alert:
                power_status['reserve_rate'] = test_reserve
            prov = _fmt_provenance(power_status, fetch_backend_json("/api/v1/dashboard/kpis"), fetch_jeju_realtime())
            with st.container(border=True, key="card_supply"):
                st.html(_html_card_head("수급 추이", "지난 12시간 · 다음 12시간", prov['supply_chips'], prov['supply_meta']))
                st.html(_html_supply_legend(power_status, prov))
                fig = create_realtime_power_chart(power_status['demand'], power_status['total_supply'],
                                                  prov['demand_measured'], prov['supply_measured'])
                st.plotly_chart(fig, theme="streamlit", width="stretch", config={"displayModeBar": False}, key="chart_supply")
                st.html(_html_time_strip(prov))
                with st.expander("표로 보기", icon=":material/table_view:"):
                    st.dataframe(_chart_table_supply(fig), hide_index=True, width="stretch")
                st.html(_html_card_foot(prov['supply_foot']))
            with st.container(key="status_row"):
                c_gauge, c_grid, c_freq = st.columns([5, 4, 3], gap="medium")
                with c_gauge:
                    with st.container(border=True, key="card_gauges"):
                        st.html(_html_card_head("수급 게이지", "", prov['demand_chips'], prov['demand_meta']))
                        g1, g2 = st.columns(2, gap="small")
                        with g1:
                            st.html(_html_gauge_label("수요", "축 0–1,200 MW · 빨간 눈금 1,080 MW"))
                            demand_gauge = create_gauge_chart(
                                value=power_status['demand'], title='', max_val=1200, color='#3B82F6', suffix=' MW'
                            )
                            st.plotly_chart(demand_gauge, theme="streamlit", width="stretch",
                                            config={"displayModeBar": False}, key="gauge_demand")
                        with g2:
                            st.html(_html_gauge_label("예비율", "빨강 10% 미만 · 호박 10–15% · 초록 15% 이상"))
                            reserve_color = '#16A34A' if power_status['reserve_rate'] >= 15 else '#B45309' if power_status['reserve_rate'] >= 10 else '#DC2626'
                            reserve_gauge = create_gauge_chart(
                                value=power_status['reserve_rate'], title='', max_val=30, color=reserve_color,
                                suffix='%', reserve=True
                            )
                            st.plotly_chart(reserve_gauge, theme="streamlit", width="stretch",
                                            config={"displayModeBar": False}, key="gauge_reserve")
                with c_grid:
                    with st.container(border=True, key="card_grid"):
                        st.html(_html_grid_state(power_status, _fmt_reserve_level(power_status['reserve_rate']), prov))
                with c_freq:
                    with st.container(border=True, key="card_freq"):
                        st.markdown(f"""<div class="pd-card-sim">
<div class="pd-card__head"><span class="pd-card__title"><span class="pd-ico pd-ico--wave" aria-hidden="true"></span>계통 주파수</span><span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>시뮬레이션 · 실측 아님</span></div>
<div class="pd-sim-value"><span class="pd-num">{power_status['frequency']:.2f}</span><span class="pd-kpi__unit">Hz</span></div>
<p class="pd-card__note">API 서버 시뮬레이션 값 (실측 아님) · 계통 실측 연동 전 참고용 · 정격 60 Hz</p>
</div>""", unsafe_allow_html=True)

        live_supply()

    with tab2:
        smp_l, smp_r = st.columns([8, 4], gap="medium")
        with smp_l:
            with st.container(border=True, key="card_smp"):
                st.html(_html_smp_head(smp_data))
                smp_view = st.segmented_control("보기", ["차트", "표"], default="차트", key="smp_view",
                                                label_visibility="collapsed") or "차트"
                if smp_view == "차트":
                    st.plotly_chart(create_smp_chart(smp_data), theme="streamlit", width="stretch",
                                    config={"displayModeBar": False}, key="chart_smp")
                else:
                    st.dataframe(_chart_table_smp(smp_data), hide_index=True, width="stretch")
                st.html(_html_card_foot(_fmt_smp_foot(smp_data)))
        with smp_r:
            with st.container(border=True, key="card_smp_stats"):
                st.html(_html_smp_stats(smp_data))
            with st.container(border=True, key="card_smp_blocks"):
                st.html(_html_smp_blocks(smp_data))
            with st.container(border=True, key="card_smp_model"):
                st.html(_html_smp_model(smp_data))
            with st.expander("예시 화면 · 입찰 예시 (실데이터 아님)", icon=":material/science:"):
                optimal_bid = smp_data['current'] * 0.95
                st.markdown(f"""<div class="pd-example">
<div class="pd-example__ribbon">예시 · 실제 값 아님</div>
<div class="pd-card__head"><span class="pd-card__title">입찰 추천</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<div class="pd-example__value"><span class="pd-num">{optimal_bid:.1f} 원/kWh</span></div>
<p class="pd-example__note">예시: 현재가에 고정 할인율 5% 적용 (최적화 결과 아님)</p>
<p class="pd-example__note">낙찰 확률 예시 (고정값, 실제 낙찰 확률 아님) <span class="pd-example__num">85%</span></p>
</div>""", unsafe_allow_html=True)

    with tab3:
        mix_col, tl_col = st.columns([5, 7], gap="medium")
        with mix_col:
            with st.container(border=True, key="card_mix"):
                st.html(_html_card_head("전력 공급 구성 (시뮬레이션 추정)", "로컬 추정 · 상단 재생에너지 비율(섬 전체 ML 추정)과 산출 근거 다름",
                                        _html_chip("est", "시뮬레이션 · 실측 아님"), ""))
                fig = create_supply_donut(power_status)
                st.plotly_chart(fig, theme="streamlit", width="stretch", config={"displayModeBar": False}, key="chart_mix")
                st.html(_html_mix_table(power_status))
        with tl_col:
            with st.container(border=True, key="card_timeline"):
                st.html(_html_card_head("시간대별 발전량 (시뮬레이션)", "등록부 27곳 기준 · 분산형 태양광 557.3 MW 제외",
                                        _html_chip("est", "시뮬레이션 · 실측 아님"), ""))
                fig = create_generation_timeline(plants_df)
                st.plotly_chart(fig, theme="streamlit", width="stretch", config={"displayModeBar": False}, key="chart_timeline")
                with st.expander("표로 보기", icon=":material/table_view:"):
                    st.dataframe(_chart_table_timeline(fig), hide_index=True, width="stretch")
        with st.container(border=True, key="card_plants"):
            st.html(_html_card_head("발전소 등록부", "위치·설비는 등록부, 출력·이용률은 시뮬레이션 추정", "", ""))
            f1, f2 = st.columns([3, 2], gap="small")
            with f1:
                type_filter = st.segmented_control("발전 유형", ["전체", "풍력", "태양광", "ESS", "화력"],
                                                   default="전체", key="plant_type") or "전체"
            with f2:
                status_filter = st.selectbox("상태", ["전체"] + sorted(plants_df['status'].dropna().unique().tolist()),
                                             key="plant_status")
            display_df = _fmt_plant_table(plants_df, type_filter, status_filter)
            st.html(_html_plant_summary(display_df))
            st.dataframe(display_df, hide_index=True, width="stretch", height=(len(display_df) + 1) * 35 + 3,
                         column_config=_fmt_plant_columns())

    with tab4:
        st.html(_html_section_head("모델 정확도와 예시 화면"))
        with st.container(border=True, key="card_model"):
            st.markdown("""<div class="pd-model">
<div class="pd-card__head"><span class="pd-card__title"><span class="pd-ico pd-ico--chart" aria-hidden="true"></span>모델 성능</span><span class="pd-chip pd-chip--measured"><span class="pd-chip__glyph" aria-hidden="true"></span>벤치마크 검증값</span></div>
<p class="pd-card__note">README 운영 모델 성능 표 기준 (2026-11-06 벤치마크 검증값)</p>
<div class="pd-model__grid">
<section class="pd-model__cell"><h4><span class="pd-ico pd-ico--plug" aria-hidden="true"></span>수요 예측 — 1시간 앞 전력수요 (KPX 제주 전력거래량)</h4><dl><dt>AR + GBM 잔차 보정 · MAPE</dt><dd class="pd-num">4.67%</dd><dt>R² Score</dt><dd class="pd-num">0.929</dd></dl><p class="pd-card__note">2025년 실측 8,760시간 (학습에 안 쓴 기간)</p></section>
<section class="pd-model__cell"><h4><span class="pd-ico pd-ico--sun" aria-hidden="true"></span>공급 추정 — 태양광 발전량 (GradientBoosting)</h4><dl><dt>KPX 오차율</dt><dd class="pd-num">17.0%</dd><dt>R² Score</dt><dd class="pd-num">0.41</dd></dl><p class="pd-card__note">holdout 2022-12~2024-05 · API 응답의 "신뢰도 93%"는 모델 성능이 아님</p></section>
<section class="pd-model__cell"><h4><span class="pd-ico pd-ico--tag" aria-hidden="true"></span>SMP 예측 — 하루전 시장가격 (어제 같은 시각 가격)</h4><dl><dt>MAPE</dt><dd class="pd-num">11.75%</dd><dt>R² Score</dt><dd class="pd-num">0.203</dd></dl><p class="pd-card__note">walk-forward 2025-07-22~12-18</p></section>
</div>
</div>""", unsafe_allow_html=True)
        with st.expander("예시 화면 3종 · 실데이터 아님", icon=":material/science:"):
            st.html(_html_example_ribbon())
            with st.container(key="ex_zone"):
                ex1, ex2, ex3 = st.columns(3, gap="medium")
                with ex1:
                    st.markdown("""<div class="pd-example">
<div class="pd-card__head"><span class="pd-card__title">XAI 분석 요약</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<p class="pd-example__lead">예시 (실제 모델 분석 결과 아님)</p>
<ul class="pd-example__list">
<li>기온 상승 (+2°C) → 수요 증가 영향 15%</li>
<li>일사량 감소 → 태양광 발전 감소 예상</li>
<li>풍속 증가 → 풍력 발전 증가 기대</li>
<li>과거 동일 시간대 패턴 반영 40%</li>
</ul>
</div>""", unsafe_allow_html=True)
                with ex2:
                    st.markdown("""<div class="pd-example">
<div class="pd-card__head"><span class="pd-card__title">리스크 알림</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<p class="pd-example__lead">예시 (실제 운영 경보 또는 입찰 권고 아님)</p>
<div class="pd-example__item"><b>덕커브 주의</b><span>14:00-16:00 태양광 발전 급증으로 SMP 하락 예상</span></div>
<div class="pd-example__item"><b>풍력 발전 증가</b><span>풍속 증가로 풍력 발전량 20% 상승 예상</span></div>
<div class="pd-example__item"><b>입찰 기회</b><span>18:00-20:00 피크 시간대 고가 입찰 권장</span></div>
</div>""", unsafe_allow_html=True)
                with ex3:
                    st.markdown("""<div class="pd-example">
<div class="pd-card__head"><span class="pd-card__title">일간 요약</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<p class="pd-example__lead">고정 예시 (실제 일간 집계 아님)</p>
<div class="pd-example__grid">
<div><span>총 발전량</span><b class="pd-num">12,450 MWh</b></div>
<div><span>CO₂ 절감</span><b class="pd-num">5,890 톤</b></div>
<div><span>평균 이용률</span><b class="pd-num">42.3%</b></div>
<div><span>예상 수익</span><b class="pd-num">₩1.2B</b></div>
</div>
</div>""", unsafe_allow_html=True)

    @st.fragment(run_every=run_every)
    def live_sources():
        st.html(_html_data_status(get_current_power_status(), get_smp_data(), get_backend_smp_predictions(),
                                  get_weather_data(), fetch_backend_json("/api/v1/dashboard/kpis"),
                                  fetch_jeju_realtime(), load_power_plants_data(), datetime.now()))

    live_sources()

    # ========== 푸터 (main() 최상위 상수: T1 '데이터 출처' 기준점) ==========
    st.markdown("""<div class="pd-footer">
<p><b>제주 전력 지도 v4.1</b> · 데이터 출처: KPX(수요·공급능력, 5분), EPSIS(SMP 하루전 가격), 초단기실황(API 서버), 발전소 등록부</p>
<p>모델: 수요 AR+GBM · 태양광 GradientBoosting · SMP 어제 같은 시각(naive) · 추정·시뮬레이션 값은 화면에 그렇게 표시합니다 · 화면 테마: 오른쪽 위 ⋮ → Settings · Power Demand Forecast Team</p>
</div>""", unsafe_allow_html=True)
