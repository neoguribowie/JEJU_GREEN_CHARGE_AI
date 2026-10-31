"""Reproducible historical policy scenario. Never modifies project inputs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import numpy as np
import pandas as pd


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results/green_match_policy_2023')
    args = parser.parse_args()
    root, out = args.project, args.out
    paths = [root / p for p in ['ai_dashboard.py', 'refresh_c005_forecast.py', 'prepare_curtailment.py',
        'data/raw/curtailment/제주 풍력 출력제어횟수 및 제어량.csv',
        'data/processed/curtailment_hourly_2021_2023.csv', 'results/curtailment_c005/predictions_2023.csv']]
    hashes = {str(p): digest(p) for p in paths}
    # Execute only the policy function and its constant, without Streamlit side effects.
    tree = ast.parse(paths[0].read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name == 'select_charging_recommendations' or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHARGING_RECOMMENDATION_THRESHOLD' for t in n.targets)]
    require(len(nodes) == 2, 'App policy definition not found')
    scope = {'pd': pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(paths[0]), 'exec'), scope)
    require(scope['CHARGING_RECOMMENDATION_THRESHOLD'] == .15, 'App threshold changed')
    keys = ['trade_date', 'trade_hour']
    pred = pd.read_csv(paths[5], parse_dates=['trade_date'])
    target = pd.read_csv(paths[4], parse_dates=['trade_date'])
    target = target[target.trade_date.dt.year.eq(2023)].copy()
    calendar = pd.MultiIndex.from_product([pd.date_range('2023-01-01', '2023-12-31'), range(1, 25)], names=keys).to_frame(index=False)
    df = calendar.merge(pred, on=keys, how='outer', validate='one_to_one', indicator=True)
    require(len(df) == 8760 and df['_merge'].eq('both').all(), 'Prediction calendar mismatch')
    df = df.drop(columns='_merge').merge(target, on=keys, how='outer', validate='one_to_one', indicator=True, suffixes=('_prediction', ''))
    require(len(df) == 8760 and df['_merge'].eq('both').all(), 'Target calendar mismatch')
    require(df.curtailment.eq(df.curtailment_prediction).all(), 'Prediction label mismatch')
    df = df.drop(columns='_merge')
    raw = pd.read_csv(paths[3], encoding='cp949')
    raw['trade_date'] = pd.to_datetime(raw['일자'], errors='raise')
    raw = raw[raw.trade_date.dt.year.eq(2023)]
    require(not raw.trade_date.duplicated().any(), 'Duplicate raw date')
    wind = raw.melt(id_vars='trade_date', value_vars=[f'{h}시' for h in range(1, 25)], var_name='hour', value_name='wind_mwh')
    wind['trade_hour'] = wind.hour.str.replace('시', '').astype(int)
    wind['wind_mwh'] = pd.to_numeric(wind.wind_mwh, errors='raise')
    require(wind.wind_mwh.notna().all() and wind.wind_mwh.ge(0).all(), 'Invalid raw amount')
    df = df.merge(wind[keys + ['wind_mwh']], on=keys, how='left', validate='one_to_one').sort_values(keys).reset_index(drop=True)
    df['source_present'] = df.wind_mwh.notna()
    df['wind_mwh'] = df.wind_mwh.fillna(0)
    require(df.wind_mwh.gt(0).astype(int).eq(df.wind_curtailment).all(), 'Wind event mismatch')
    df['forecast_datetime'] = df.trade_date + pd.to_timedelta(df.trade_hour, unit='h')
    require(pd.to_datetime(df.timestamp).eq(df.forecast_datetime).all(), 'Timestamp mismatch')
    df['risk_score'] = pd.to_numeric(df.probability, errors='raise')
    require(df.risk_score.between(0, 1).all(), 'Invalid score')
    selected = scope['select_charging_recommendations'](df)
    df['recommended'] = df.forecast_datetime.isin(selected.forecast_datetime)
    # Independent check of ordering, threshold and tie break.
    expected = df[df.trade_hour.between(10, 16) & df.risk_score.ge(.15)].sort_values(['trade_date', 'risk_score', 'forecast_datetime'], ascending=[True, False, True]).groupby('trade_date').head(3)
    require(set(expected.forecast_datetime) == set(selected.forecast_datetime), 'App policy parity failed')
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'joined_hourly_2023.csv', index=False, encoding='utf-8-sig')
    selected.to_csv(out / 'recommendations_2023.csv', index=False, encoding='utf-8-sig')
    counts = df.groupby('trade_date').recommended.sum()
    assumptions = dict(power_kw=50, grid_kwh_per_vehicle=20, service_minutes=24, turnover_minutes=6,
        vehicles_per_day=[100, 500, 1000], vehicles_per_charger=[10, 14, 5], main_vehicles_per_charger=10,
        daytime_trade_hours=list(range(10, 17)), physical_daytime='09:00–16:00 KST',
        night_trade_hours=list(range(1, 8)), physical_night='00:00–07:00 KST on same trade date',
        fallback='00:00–07:00 separate uncredited charging; same daily demand accounting',
        emission_factors=[.3, .4, .5], missing_raw='zero as existing preprocessing; not verified observed zero',
        wind_unit='MWh per previous user confirmation, not independently established by filename')
    summaries, daily_rows, hourly_rows = [], [], []
    for ratio in [10, 14, 5]:
        for n in [100, 500, 1000]:
            chargers = math.ceil(n / ratio)
            capacity = chargers * 2  # full sessions per hour, each 24+6 minutes
            demand = n * .02
            for strategy in ['night', 'day_uniform', 'ai']:
                total_recovered = total_scheduled = total_fallback = 0.
                for day, group in df.groupby('trade_date', sort=True):
                    hours = list(range(1, 8)) if strategy == 'night' else list(range(10, 17)) if strategy == 'day_uniform' else sorted(group.loc[group.recommended, 'trade_hour'].tolist())
                    # Whole vehicles, equal allocation within one vehicle; no use of actual wind in dispatch.
                    allocation = {h: 0 for h in range(1, 25)}
                    remaining = n
                    while hours and remaining and any(allocation[h] < capacity for h in hours):
                        for h in hours:
                            if remaining and allocation[h] < capacity:
                                allocation[h] += 1
                                remaining -= 1
                    fallback = {h: 0 for h in range(1, 8)}
                    for vehicle in range(remaining):
                        fallback[vehicle % 7 + 1] += 1
                    require(max(fallback.values()) <= capacity, 'Fallback fleet exceeds charger capacity')
                    recovered_day = scheduled_day = 0.
                    for row in group.itertuples():
                        vehicles = allocation[row.trade_hour]
                        energy = vehicles * .02
                        recovery = min(energy, row.wind_mwh)
                        fb = fallback.get(row.trade_hour, 0)
                        require(vehicles + fb <= capacity, 'Hourly charger capacity exceeded')
                        require(recovery <= row.wind_mwh and recovery <= energy, 'Recovery bound failed')
                        hourly_rows.append([ratio, n, chargers, strategy, day, row.trade_hour, vehicles, fb, energy, fb*.02, row.wind_mwh, recovery])
                        recovered_day += recovery
                        scheduled_day += energy
                    require(np.isclose(scheduled_day + remaining*.02, demand), 'Demand conservation failed')
                    if strategy == 'ai' and not hours:
                        require(recovered_day == 0, 'No recommendation credit must be zero')
                    daily_rows.append([ratio, n, strategy, day, len(hours), scheduled_day, remaining*.02, recovered_day])
                    total_recovered += recovered_day
                    total_scheduled += scheduled_day
                    total_fallback += remaining*.02
                summaries.append(dict(vehicles_per_charger=ratio, vehicles_per_day=n, chargers=chargers, strategy=strategy,
                    annual_demand_mwh=demand*365, scheduled_mwh=total_scheduled, fallback_mwh=total_fallback,
                    potential_recovery_upper_mwh=total_recovered))
    summary = pd.DataFrame(summaries)
    effects = []
    for (ratio, n), group in summary.groupby(['vehicles_per_charger', 'vehicles_per_day']):
        v = group.set_index('strategy').potential_recovery_upper_mwh
        for label, value in [('night_to_day', v.day_uniform-v.night), ('day_to_ai', v.ai-v.day_uniform), ('night_to_ai', v.ai-v.night)]:
            effects.append(dict(vehicles_per_charger=ratio, vehicles_per_day=n, effect=label, delta_upper_mwh=value,
                **{f'conditional_tco2_ef_{ef}': value*ef for ef in [.3, .4, .5]}))
    effects = pd.DataFrame(effects)
    for ef in [.3, .4, .5]:
        summary[f'conditional_tco2_ef_{ef}'] = summary.potential_recovery_upper_mwh*ef
    hourly = pd.DataFrame(hourly_rows, columns=['vehicles_per_charger','vehicles_per_day','chargers','strategy','trade_date','trade_hour','scheduled_vehicles','fallback_vehicles','scheduled_mwh','fallback_mwh','actual_wind_mwh','potential_recovery_upper_mwh'])
    daily = pd.DataFrame(daily_rows, columns=['vehicles_per_charger','vehicles_per_day','strategy','trade_date','charging_hours_count','scheduled_mwh','fallback_mwh','potential_recovery_upper_mwh'])
    require(np.allclose(hourly.groupby(['vehicles_per_charger','vehicles_per_day','strategy']).potential_recovery_upper_mwh.sum().sort_index(), summary.set_index(['vehicles_per_charger','vehicles_per_day','strategy']).potential_recovery_upper_mwh.sort_index()), 'Aggregation mismatch')
    for name, frame in [('scenario_summary',summary),('effect_decomposition',effects),('simulation_hourly_2023',hourly),('simulation_daily_2023',daily)]:
        frame.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    hourly.groupby(['vehicles_per_charger','vehicles_per_day','strategy',pd.to_datetime(hourly.trade_date).dt.month])[['scheduled_mwh','fallback_mwh','potential_recovery_upper_mwh']].sum().to_csv(out/'monthly_summary.csv', encoding='utf-8-sig')
    require(all(digest(Path(p)) == h for p,h in hashes.items()), 'Input file modified during run')
    validation = dict(joined_rows=len(df), prediction_target_label_mismatches=0, wind_event_mismatches=0,
        wind_mwh=float(df.wind_mwh.sum()), raw_missing_hours=int((~df.source_present).sum()),
        recommendation_hours=len(selected), no_recommendation_days=int(counts.eq(0).sum()),
        recommendation_day_counts={str(k):int(v) for k,v in counts.value_counts().sort_index().items()},
        app_function_parity=True, hourly_capacity_and_daily_demand_checks=True, input_sha256=hashes)
    (out/'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'assumptions.json').write_text(json.dumps(assumptions, ensure_ascii=False, indent=2), encoding='utf-8')
    def table(frame):
        return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.round(3).itertuples(index=False, name=None))
    main_summary = summary[summary.vehicles_per_charger.eq(10)]
    main_effects = effects[effects.vehicles_per_charger.eq(10)].drop(columns='vehicles_per_charger')
    report = f'''# Green Match 2023 추천 정책 재현 시나리오

## 먼저 알아둘 결론
아래 수치는 실제 출력제어 완화나 실증 CO2 감축량이 아니다. 시간별 잠재 회수량=min(가정한 추가 충전 MWh, 과거 실제 풍력 출력제어 MWh)으로 계산한 **이론적 상한**이다. 점수는 시간 선택에만 사용하며 발전량으로 환산하지 않는다. 음수 증분도 그대로 표시한다.

주 분석에서는 AI 전략의 연간 회수 상한이 낮 균등보다 낮았다. 추천일·추천시간이 제한되고 충전기 처리용량을 초과한 수요를 회수 실적 없는 대체 충전으로 넘기기 때문이다. 따라서 이번 가정에서 AI의 추가 효과가 양수라는 주장은 지지되지 않는다. 야간 대비 증가분과 낮 균등 대비 감소분을 구분해서 읽어야 한다.

## 앱과 시간 기준
ai_dashboard.py의 select_charging_recommendations 함수와 임계값을 AST로 직접 읽어 실행했다. 확률 probability를 risk_score로 사용하고, 종료시각 forecast_datetime=trade_date+trade_hour로 복원했다. 점수 0.15 이상, 종료시각 10~16시 중 점수 내림차순·동점은 시간 오름차순으로 날짜별 최대 3개를 추천한다. prediction_selected(분류 임계값 0.05)는 사용하지 않는다.

prepare_curtailment.py 및 refresh_c005_forecast.py는 종료 기준 거래시간을 사용한다. 따라서 앱의 10~16시는 거래시간 10~16, 실제 구간 09~16시다. 이전 보고서의 10~17시 충전(거래시간 11~17)과 한 시간 차이가 있다. 앱과 일치하도록 본 분석을 수정했다. 이것이 제품 의도와 다르다면 앱 자체의 시간 정의를 별도로 수정해야 한다.

## 결합 검증
2023년 전체 8,760행을 달력·예측·정답과 일대일 결합했다. 미매칭·중복·분류 정답 불일치·풍력 발생 플래그 불일치 0건이며 시간 timestamp도 일치한다. 풍력 합계 {df.wind_mwh.sum():,.0f} MWh. 원시 미기록 {validation['raw_missing_hours']:,}시간은 기존 전처리와 같이 0 처리했다. 이는 실측 0을 확인한 것이 아니며 source_present로 구분했다. MWh 단위는 이전 사용자 확인을 따르며 파일명만으로 독립 검증되지는 않는다.
추천 {len(selected)}시간, 추천 0개인 날 {counts.eq(0).sum()}일. 날짜별 추천 개수 분포: {validation['recommendation_day_counts']}. 원본 함수 실행 결과와 독립 정렬 구현의 선택 결과가 일치했다. validation.json에 입력 SHA256과 검증 결과를 저장했다.

## 차량·충전기·체류 가정
매일 100/500/1,000대, 대당 계통 입력 20kWh, 평균 50kW 급속 충전을 적용한다. 1대 충전 24분+교체 6분으로 충전기당 시간당 최대 2대, 실제 최대 동시 충전은 충전기 수 이하다. 주 분석 충전기는 차량 수/10 올림으로 10/50/100기, 전력 설비 상한은 0.5/2.5/5MW다. 차량당 해당 시간 슬롯에 최소 30분 체류하고 예약·순차 입차가 가능하다는 가정이다. 전 차량이 동시에 충전한다는 가정은 하지 않았다. 충전 속도 저하·고장·대기·이동·배터리 SOC 제약은 별도 실측하지 않았다.

충전기당 하루 참여 차량 14대(8/36/72기), 5대(20/100/200기)의 민감도 결과도 CSV에 포함한다. 추가 충전기 공급은 실제 확보를 의미하지 않는다. 7kW 결과는 이전 보고서에만 존재하며 이번 급속 시나리오와 직접 섞지 않는다.

야간 기준선은 같은 거래일 00~07시(거래시간 1~7)에 전체 차량을 7시간 균등 배분한다. 낮 균등은 앱 후보와 같은 09~16시(거래시간 10~16)에 배분한다. 차량은 정수로 배정해 시간별 차이는 최대 1대다. AI는 추천 시간에 균등 배분하되 충전기 처리용량을 넘기지 않는다. 남은 차량은 별도 00~07시 대체 충전으로 배분하며 회수 실적 0이다. 이는 하루 수요 회계 기준의 사전 예약 스케줄이며, 추천이 당일 늦게 나올 때 이미 지난 야간에 소급 충전할 수 있다는 의미가 아니다. 실운영에는 전날 예약 또는 다음날 야간 이월 검증이 필요하다.

추천 0개인 날은 전량 대체 충전하며 회수 실적 0이다. 추천 부족·용량 부족 잔여 수요도 동일하다. 대체량까지 매일 총수요를 보존하고 시간별 충전기 제약을 검사했다. 야간 기준선 자체의 풍력 회수 상한은 야간 실제 제어량과 계산하므로 임의로 0으로 놓지 않았다.

## 주 분석 결과
strategy: night=야간, day_uniform=낮 균등, ai=앱 추천. scheduled_mwh는 해당 전략 시간 배정, fallback_mwh는 회수 실적 없는 대체 충전이다. potential_recovery_upper_mwh는 연간 이론적 상한이며 conditional_tco2는 조건부 저감 환산량이다.

{table(main_summary.drop(columns=['vehicles_per_charger','annual_demand_mwh']))}

## 효과 분리
night_to_day=야간→낮 이동 효과(낮 균등−야간). day_to_ai=낮 균등→AI 시간 최적화 효과(AI−낮 균등). night_to_ai는 두 효과의 합이다. 음수이면 해당 가정에서 상한이 감소했다는 뜻이다. 추천 시간 수·충전기 제약·대체 충전 정책까지 포함한 전략 차이이며 순수한 순위 품질의 인과효과는 아니다.

{table(main_effects)}

## 해석의 한계
배출계수 0.3/0.4/0.5 tCO2/MWh는 민감도 가정이다. 상한 MWh×계수 또는 전략 간 상한 차이×계수를 **조건부 저감 환산량**으로 표시했다. 실제 한계배출계수 없는 실증 CO2 감축량이 아니다. 같은 수요의 기존 계통전력을 대체하고 추가 충전이 실제 제어 완화로 이어져야 한다. 제주 전체 제어량과 충전기의 위치·계통 제약이 일치한다는 보장이 없어 이 상한은 낙관적이다. 기준선의 충전도 '추가 수요'라는 가상 비교 조건을 공유하며 관측 기준선이 아니다.

과거 실제 기상 기반 예측 평가로 당시 예보·추천 공지 시각·실시간 가용성을 재현하지 않는다. 원시 미기록=0, 충전기 공급, 모든 날 동일 참여, 예약 준수 가정에 민감하다. 태양광 MWh는 미확보라 제외했다. 입력 원본·학습·예측 데이터는 수정하지 않았다.

## 재실행 및 산출물
스크립트 기본 경로는 원래 프로젝트를 읽고 프로젝트 results/green_match_policy_2023에만 쓴다. 예: 프로젝트 .venv/Scripts/python.exe green_match_simulation.py --project "프로젝트 경로" --out "결과 경로". pandas와 numpy가 필요하다. ai_dashboard.py의 추천 함수 변경 시 다시 읽어 실행하고 임계값 변경은 중단한다.

joined_hourly_2023.csv는 결합·출처·선택 검증용, recommendations_2023.csv는 실제 정책 선택, simulation_hourly_2023.csv는 차량·대체 충전·에너지·회수 상한, simulation_daily_2023.csv와 monthly_summary.csv는 집계, scenario_summary.csv는 충전기 민감도 포함 연간 결과, effect_decomposition.csv는 효과 분리와 배출계수별 환산량이다. assumptions.json과 validation.json으로 가정·입력 해시를 확인할 수 있다. 기존 스크립트는 green_match_simulation_before_policy_fix.py로 보존했다.
'''
    (out/'report.md').write_text(report, encoding='utf-8')
    print(main_summary.round(3).to_string(index=False))
    print(main_effects.round(3).to_string(index=False))
    print('Verified outputs:', out)


if __name__ == '__main__':
    main()
