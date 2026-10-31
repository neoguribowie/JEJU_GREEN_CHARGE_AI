"""Safe charger diagnostics. Never include request URLs or exception text."""
import json
import re
import requests


class ChargerAPIError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


def provider_error(code):
    # Only documented-style numeric codes may be reflected from remote data.
    value = str(code).strip()
    value = value if re.fullmatch(r'\d{1,4}', value) else 'UNKNOWN'
    messages = {
        '03': 'API에서 조회할 데이터가 없다고 응답했습니다.',
        '10': '충전소 API 요청 항목을 확인해 주세요.',
        '20': '충전소 API 이용 권한을 확인해 주세요.',
        '22': '충전소 API 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.',
        '30': '충전소 API 인증키의 등록 상태를 확인해 주세요.',
        '31': '충전소 API 인증키의 사용 기간을 확인해 주세요.',
        '32': '충전소 API에 등록된 접근 IP를 확인해 주세요.',
    }
    return ChargerAPIError('API_'+value, messages.get(value,'충전소 API가 오류를 반환했습니다. 공공데이터포털의 API 이용 상태를 확인해 주세요.'))


def describe_charger_error(error, stage='조회'):
    if isinstance(error, ChargerAPIError):
        return error.code, error.message
    if isinstance(error, requests.exceptions.ProxyError):
        return 'PROXY_ERROR', '프록시 연결에 실패했습니다. 앱을 실행한 터미널의 네트워크 환경을 확인하거나 일반 터미널에서 다시 실행해 주세요.'
    if isinstance(error, requests.exceptions.SSLError):
        return 'SSL_ERROR', 'API 연결의 인증서를 확인하지 못했습니다. 네트워크와 인증서 설정을 확인해 주세요.'
    if isinstance(error, requests.exceptions.Timeout):
        return 'TIMEOUT', '충전소 API 응답 시간이 초과되었습니다. 잠시 후 다시 검색해 주세요.'
    if isinstance(error, requests.exceptions.HTTPError):
        status = getattr(getattr(error,'response',None),'status_code',None)
        return 'HTTP_'+str(status if isinstance(status,int) else 'UNKNOWN'), '충전소 API 서버가 HTTP 오류를 반환했습니다. 잠시 후 다시 시도해 주세요.'
    if isinstance(error, requests.exceptions.ConnectionError):
        return 'CONNECTION_ERROR', '충전소 API에 연결하지 못했습니다. 인터넷 연결 상태를 확인해 주세요.'
    if isinstance(error, requests.exceptions.RequestException):
        return 'REQUEST_ERROR', '충전소 API 요청을 완료하지 못했습니다.'
    if isinstance(error, json.JSONDecodeError):
        return 'SNAPSHOT_JSON_ERROR', '저장된 시연 데이터의 JSON 형식이 올바르지 않습니다. 실시간 조회로 다시 저장해 주세요.'
    if isinstance(error, OSError):
        number = getattr(error,'errno',None)
        suffix = '_'+str(number) if isinstance(number,int) else ''
        code = 'SNAPSHOT_WRITE_ERROR' if stage == '시연 데이터 저장' else 'FILE_ERROR'
        return code+suffix, '데이터 파일에 접근하지 못했습니다. 파일 권한이나 다른 프로그램의 사용 여부를 확인해 주세요.'
    if isinstance(error,(KeyError,ValueError,TypeError,AttributeError)):
        return 'DATA_'+type(error).__name__.upper(), '충전소 데이터 형식이 예상과 다릅니다. 실시간 조회 또는 시연 데이터 재저장이 필요합니다.'
    return 'UNEXPECTED_ERROR', '예상하지 못한 충전소 처리 오류가 발생했습니다.'
