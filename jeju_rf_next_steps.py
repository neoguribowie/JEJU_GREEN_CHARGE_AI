"""Observed-weather baseline. See jeju_weather_review.md before training."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True,
                   help='Folder containing the three OBS_ASOS_TIM_YEAR.csv files')
    p.add_argument('--wind', type=Path, required=True)
    p.add_argument('--alignment', choices=['start', 'end'], required=True,
                   help='start: date+(hour-1); end: date+hour. Confirm source convention.')
    p.add_argument('--output', type=Path, default=Path('rf_outputs'))
    p.add_argument('--train', action='store_true')
    p.add_argument('--target-definition-confirmed', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    weather = pd.concat([
        pd.read_csv(args.data / f'OBS_ASOS_TIM_{y}.csv', encoding='cp949')
        for y in (2023, 2024, 2025)
    ], ignore_index=True)
    weather['timestamp'] = pd.to_datetime(weather['일시'], errors='raise')
    if weather.duplicated(['지점', 'timestamp']).any():
        raise ValueError('Duplicate station/timestamp keys')
    if set(weather['지점']) != {184, 185, 188, 189}:
        raise ValueError('Unexpected station set')
    mapping = {
        '기온(°C)': ('temp', '기온 QC플래그'),
        '풍속(m/s)': ('ws', '풍속 QC플래그'),
        '풍향(16방위)': ('wd', '풍향 QC플래그'),
        '습도(%)': ('rh', '습도 QC플래그'),
        '현지기압(hPa)': ('pressure', '현지기압 QC플래그'),
    }
    for col, (name, qc) in mapping.items():
        weather[name] = pd.to_numeric(weather[col], errors='raise')
        # Blank QC is retained; explicit error/missing QC is masked.
        weather.loc[pd.to_numeric(weather[qc], errors='coerce').isin([1, 9]), name] = np.nan
    if not weather['wd'].dropna().between(0, 360).all():
        raise ValueError('Unexpected wind direction units')
    # CSV values are degrees despite the 16-direction column label.
    theta = np.deg2rad(weather['wd'])
    weather['u'] = -weather['ws'] * np.sin(theta)
    weather['v'] = -weather['ws'] * np.cos(theta)
    # A zero-speed observation has zero components even if direction is absent.
    weather.loc[weather['ws'].eq(0), ['u', 'v']] = 0.0
    names = ['temp', 'ws', 'rh', 'pressure', 'u', 'v']
    wide = weather.pivot(index='timestamp', columns='지점', values=names)
    wide.columns = [f'{name}_{station}' for name, station in wide.columns]
    wide['weather_key_present'] = True
    wind = pd.read_csv(args.wind, encoding='utf-8-sig')
    wind['trade_hour'] = pd.to_numeric(wind['trade_hour'], errors='raise')
    if not wind['trade_hour'].isin(range(1, 25)).all():
        raise ValueError('Trade hours must be integers 1..24')
    if wind.duplicated(['trade_date', 'trade_hour']).any():
        raise ValueError('Duplicate wind date/hour keys')
    wind['timestamp'] = pd.to_datetime(wind['trade_date']) + pd.to_timedelta(
        wind['trade_hour'] - (args.alignment == 'start'), unit='h')
    wind['wind_mwh'] = pd.to_numeric(wind['wind_mwh'], errors='raise')
    merged = wind.merge(wide, on='timestamp', how='left', validate='one_to_one').sort_values('timestamp')
    merged.to_csv(args.output / 'wind_weather_observed.csv', index=False, encoding='utf-8-sig')
    summary = {'rows': len(merged), 'alignment': args.alignment,
               'unmatched_weather_hours': int(merged.weather_key_present.isna().sum()),
               'missing_target': int(merged.wind_mwh.isna().sum()),
               'feature_missing_after_qc': merged[wide.columns[:-1]].isna().sum().to_dict()}
    (args.output / 'merge_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if not args.train:
        return
    if not args.target_definition_confirmed:
        raise ValueError('Confirm target units/population across source revisions before --train')
    if summary['unmatched_weather_hours']:
        raise ValueError('Resolve unmatched weather hours before training')
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
    X = merged[list(wide.columns[:-1])].copy()
    ts = merged.timestamp
    X['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
    X['season_sin'] = np.sin(2 * np.pi * (ts.dt.dayofyear - 1) / 365.2425)
    X['season_cos'] = np.cos(2 * np.pi * (ts.dt.dayofyear - 1) / 365.2425)
    # Split by original trade date so hour 24 remains in its source year.
    years = pd.to_datetime(merged.trade_date).dt.year
    valid = merged.wind_mwh.notna()
    train = valid & years.eq(2023)
    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                          RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                max_features=0.8, random_state=42, n_jobs=-1))
    baseline = DummyRegressor(strategy='mean')
    model.fit(X[train], merged.loc[train, 'wind_mwh'])
    baseline.fit(X[train], merged.loc[train, 'wind_mwh'])
    records = []
    for year, label in [(2024, 'validation'), (2025, 'test')]:
        use = valid & years.eq(year)
        y = merged.loc[use, 'wind_mwh']
        for name, estimator in [('rf', model), ('train_mean', baseline)]:
            pred = estimator.predict(X[use])
            records.append({'split': label, 'model': name, 'n': len(y),
                            'MAE': mean_absolute_error(y, pred),
                            'RMSE': np.sqrt(mean_squared_error(y, pred)), 'R2': r2_score(y, pred)})
        pd.DataFrame({'timestamp': ts[use], 'actual': y, 'prediction': model.predict(X[use])}).to_csv(
            args.output / f'predictions_{year}.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(args.output / 'metrics.csv', index=False, encoding='utf-8-sig')
    joblib.dump({'pipeline': model, 'features': X.columns.tolist(),
                 'alignment': args.alignment, 'training_year': 2023}, args.output / 'rf_observed.joblib')


if __name__ == '__main__':
    main()
