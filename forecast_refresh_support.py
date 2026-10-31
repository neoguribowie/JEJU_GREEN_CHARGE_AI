"""Run integrated forecasts with safe diagnostics and failure rollback."""
from pathlib import Path
import subprocess
import re
import threading

_REFRESH_LOCK = threading.Lock()
OUTPUTS = (
    'data/processed/jeju_weather_forecast_4stations.csv',
    'results/curtailment_c005/future_risk_forecast.csv',
    'results/demand_d002/future_demand_forecast.csv',
    'results/curtailment_c006/future_risk_forecast.csv',
)


class ForecastRefreshError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


def diagnose_refresh_output(stderr):
    # This function returns only fixed explanations and numeric service codes.
    # Never return the raw exception message, URL, response body or credential.
    text = stderr.decode('utf-8', errors='replace') if isinstance(stderr, bytes) else str(stderr or '')
    if 'ProxyError' in text:
        return ForecastRefreshError('PROXY_ERROR', '프록시 연결에 실패했습니다. 앱을 실행한 터미널의 네트워크 환경을 확인해 주세요.')
    if 'SSLError' in text:
        return ForecastRefreshError('SSL_ERROR', '기상청 API 연결의 인증서를 확인하지 못했습니다. 네트워크와 인증서 설정을 확인해 주세요.')
    if any(word in text for word in ('ReadTimeout', 'ConnectTimeout', 'TimeoutError')):
        return ForecastRefreshError('API_TIMEOUT', '기상청 API 응답이 지연되었습니다. 잠시 후 다시 갱신해 주세요.')
    if 'ConnectionError' in text:
        return ForecastRefreshError('CONNECTION_ERROR', '기상청 API에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요.')
    http = re.findall(r'\b([45]\d\d)\s+(?:Client|Server)\s+Error', text)
    if http:
        return ForecastRefreshError('HTTP_'+http[-1], '기상청 API가 HTTP 오류를 반환했습니다. 잠시 후 다시 갱신해 주세요.')
    api = re.findall(r'''['"]resultCode['"]\s*:\s*['"](\d{1,4})['"]''', text)
    if api:
        code = api[-1]
        hint = {
            '03': '선택한 발표 시각의 예보가 아직 제공되지 않습니다. 잠시 후 다시 갱신해 주세요.',
            '22': '기상청 API 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.',
            '30': '기상청 API 인증키의 등록 상태를 확인해 주세요.',
            '31': '기상청 API 인증키의 사용 기간을 확인해 주세요.',
        }.get(code, '기상청 API가 오류를 반환했습니다. API 이용 상태를 확인해 주세요.')
        return ForecastRefreshError('KMA_API_'+code, hint)
    if 'JSONDecodeError' in text or 'JSON 응답' in text:
        return ForecastRefreshError('KMA_RESPONSE_FORMAT', '기상청 API가 예상한 JSON 응답을 반환하지 않았습니다. 잠시 후 다시 갱신해 주세요.')
    if 'PermissionError' in text:
        return ForecastRefreshError('FILE_PERMISSION', '예측 파일을 저장하지 못했습니다. CSV를 사용하는 프로그램이나 파일 권한을 확인해 주세요.')
    if 'FileNotFoundError' in text:
        return ForecastRefreshError('FILE_NOT_FOUND', '갱신에 필요한 모델 또는 데이터 파일을 찾지 못했습니다.')
    if 'StreamlitSecretNotFoundError' in text or 'KMA_API_KEY' in text and 'KeyError' in text:
        return ForecastRefreshError('KMA_KEY_CONFIG', '기상청 API 인증 설정을 확인해 주세요.')
    if 'ModuleNotFoundError' in text or 'ImportError' in text:
        return ForecastRefreshError('DEPENDENCY_ERROR', '앱의 가상환경과 필수 라이브러리 설치 상태를 확인해 주세요.')
    if 'UnicodeEncodeError' in text:
        return ForecastRefreshError('ENCODING_ERROR', '예측 갱신 프로그램의 출력 인코딩을 처리하지 못했습니다.')
    if any(word in text for word in ('예보값 누락', '입력변수 누락', '응답 일부만 수집됨', '기상 입력변수')):
        return ForecastRefreshError('FORECAST_DATA_ERROR', '예측에 필요한 기상 자료가 완전하지 않습니다. 잠시 후 다시 갱신해 주세요.')
    return ForecastRefreshError('FORECAST_PROCESS_ERROR', '통합 예측 프로그램이 정상 종료되지 않았습니다. 실행 환경과 모델 입력 자료를 확인해 주세요.')


def run_integrated_refresh(root, executable, timeout=180):
    root = Path(root)
    if not _REFRESH_LOCK.acquire(blocking=False):
        raise ForecastRefreshError('REFRESH_BUSY', '다른 예측 갱신이 진행 중입니다. 완료 후 다시 시도해 주세요.')
    snapshots = {}
    changed = False
    try:
        # Do not start a refresh unless all previous outputs can be backed up.
        for rel in OUTPUTS:
            path = root / rel
            snapshots[path] = path.read_bytes() if path.exists() else None
        changed = True
        result = subprocess.run(
            [executable, '-X', 'utf8', str(root / 'refresh_all_forecasts.py')],
            cwd=str(root), capture_output=True, timeout=timeout,
        )
        if result.returncode:
            raise diagnose_refresh_output(result.stderr)
        # Success means all model results refer to the same issuance and times.
        import pandas as pd
        frames = [pd.read_csv(root / rel) for rel in OUTPUTS[1:]]
        reference = frames[0]
        issued = set(reference['forecast_issued_at'].astype(str))
        times = set(reference['forecast_datetime'].astype(str))
        if len(issued) != 1 or not times:
            raise ForecastRefreshError('FORECAST_ALIGNMENT', '갱신 결과의 예보 기준 또는 예측 시점을 확인하지 못했습니다.')
        for frame in frames:
            if set(frame['forecast_issued_at'].astype(str)) != issued or set(frame['forecast_datetime'].astype(str)) != times or len(frame) != len(times):
                raise ForecastRefreshError('FORECAST_ALIGNMENT', '모델별 예보 기준 또는 예측 시점이 일치하지 않아 갱신을 취소했습니다.')
    except Exception as error:
        if changed:
            try:
                for path, content in snapshots.items():
                    if content is None:
                        if path.exists(): path.unlink()
                    elif not path.exists() or path.read_bytes() != content:
                        path.write_bytes(content)
            except OSError:
                raise ForecastRefreshError('ROLLBACK_FAILED', '기존 예측 복원 중 파일 접근 오류가 발생했습니다. CSV 파일의 사용 여부와 모델별 예보 기준을 확인해 주세요.') from None
        if isinstance(error, ForecastRefreshError):
            raise
        if isinstance(error, subprocess.TimeoutExpired):
            raise ForecastRefreshError('REFRESH_TIMEOUT', '통합 예측 갱신 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.') from None
        if isinstance(error, OSError):
            raise ForecastRefreshError('FILE_ACCESS_ERROR', '갱신 프로그램 실행 또는 결과 파일 접근에 실패했습니다. 파일 권한과 가상환경을 확인해 주세요.') from None
        raise ForecastRefreshError('FORECAST_DATA_ERROR', '예측 결과 확인 중 오류가 발생했습니다. 모델 입력 자료를 확인해 주세요.') from None
    finally:
        _REFRESH_LOCK.release()
