"""
MAC(Multiply-Accumulate) 기반 패턴 판별 프로그램

- 모드 1: 3×3 필터 A, B와 패턴을 콘솔로 입력받아 판정
- 모드 2: data.json의 필터/패턴(5×5, 13×13, 25×25)을 일괄 판정
- 성능 분석: 크기별 MAC 연산 시간(10회 평균)과 연산 횟수(N²) 표 출력

외부 라이브러리 없이 표준 라이브러리(json, time, os)만 사용한다.
Python 3.8 이상에서 동작한다.
"""

import json
import os
import time

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
EPSILON = 1e-9          # 두 점수의 차이가 이 값보다 작으면 동점으로 본다
REPEAT = 10             # 성능 측정 시 반복 횟수
PERF_SIZES = [3, 5, 13, 25]
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

# 표준 라벨 (프로그램 내부에서는 이 두 가지만 사용한다)
LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"


# ---------------------------------------------------------------------------
# 1. 데이터 구조: n×n 2차원 배열(리스트의 리스트)
# ---------------------------------------------------------------------------
def create_matrix(n, fill=0.0):
    """n×n 크기의 2차원 배열을 만든다."""
    return [[fill for _ in range(n)] for _ in range(n)]


def set_value(matrix, row, col, value):
    """특정 위치(row, col)에 값을 저장한다."""
    matrix[row][col] = value


def get_value(matrix, row, col):
    """특정 위치(row, col)의 값을 읽어온다."""
    return matrix[row][col]


def is_square(matrix, n):
    """matrix가 n×n 크기인지 검사한다. (스키마 검증용)"""
    if not isinstance(matrix, list) or len(matrix) != n:
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            return False
    return True


# ---------------------------------------------------------------------------
# 2. 라벨 정규화
#    data.json에는 같은 의미의 라벨이 '+', 'cross', 'x' 등 여러 표기로 섞여 있다.
#    비교 전에 모두 표준 라벨(Cross / X)로 바꿔야 PASS/FAIL이 올바르게 나온다.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "+": LABEL_CROSS,       # expected 값
    "cross": LABEL_CROSS,   # filter 키
    "x": LABEL_X,           # expected 값, filter 키 공통
}


def normalize_label(raw):
    """원본 라벨을 표준 라벨로 바꾼다. 알 수 없는 라벨이면 None을 돌려준다."""
    if not isinstance(raw, str):
        return None
    return LABEL_MAP.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# 3. MAC 연산 (반복문으로 직접 구현)
#    같은 위치의 값끼리 곱한 뒤 모두 더한다 → 패턴과 필터가 얼마나 겹치는지(유사도)
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """2차원 패턴과 필터의 MAC 점수를 계산한다. 연산 횟수는 N×N번."""
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]   # Multiply → Accumulate
    return total


# [보너스] 1차원 배열 버전 -------------------------------------------------
def flatten(matrix):
    """2차원 배열을 길이 N²의 1차원 배열로 바꾼다."""
    flat = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def mac_1d(pattern_flat, filt_flat):
    """1차원 배열끼리 MAC 연산. 인덱스 하나로만 접근하므로 접근이 단순하다."""
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filt_flat[k]
    return total


# ---------------------------------------------------------------------------
# 4. 점수 비교 (epsilon 기반)
#    부동소수점은 0.9와 0.8999999999999999처럼 미세한 오차가 생긴다.
#    단순히 a > b로 비교하면 사실상 동점인데도 한쪽이 이긴 것으로 잘못 판정된다.
# ---------------------------------------------------------------------------
def compare_scores(score_a, score_b, label_a, label_b, tie_label):
    """두 점수를 비교해 더 큰 쪽의 라벨을 돌려준다. 차이가 EPSILON 미만이면 tie_label."""
    if abs(score_a - score_b) < EPSILON:
        return tie_label
    if score_a > score_b:
        return label_a
    return label_b


# ---------------------------------------------------------------------------
# 5. 성능 측정
#    입력/출력/파일 읽기는 빼고 "연산 함수 호출 구간"만 잰다.
# ---------------------------------------------------------------------------
def measure_avg_ms(func, a, b, repeat=REPEAT):
    """func(a, b)를 repeat번 실행한 평균 시간(ms)을 돌려준다."""
    start = time.perf_counter()
    for _ in range(repeat):
        func(a, b)
    end = time.perf_counter()
    return (end - start) * 1000 / repeat


# [보너스] 패턴 생성기 -------------------------------------------------------
def generate_cross(n):
    """N×N 십자가(+) 패턴을 만든다. (가운데 행과 가운데 열이 1)"""
    matrix = create_matrix(n)
    mid = n // 2
    for i in range(n):
        set_value(matrix, mid, i, 1.0)
        set_value(matrix, i, mid, 1.0)
    return matrix


def generate_x(n):
    """N×N X 패턴을 만든다. (두 대각선이 1)"""
    matrix = create_matrix(n)
    for i in range(n):
        set_value(matrix, i, i, 1.0)
        set_value(matrix, i, n - 1 - i, 1.0)
    return matrix


def print_performance_table(sizes):
    """크기별 평균 연산 시간과 연산 횟수(N²) 표를 출력한다."""
    print()
    print("[성능 분석] 크기별 MAC 연산 시간 (반복 {}회 평균)".format(REPEAT))
    print("-" * 64)
    print("{:<10}{:>16}{:>18}{:>16}".format("크기", "2차원 평균(ms)", "1차원 평균(ms)", "연산 횟수(N²)"))
    print("-" * 64)
    for n in sizes:
        pattern = generate_cross(n)   # 측정용 입력은 생성기로 만든다(같은 입력, 같은 반복)
        filt = generate_x(n)
        avg_2d = measure_avg_ms(mac, pattern, filt)
        avg_1d = measure_avg_ms(mac_1d, flatten(pattern), flatten(filt))
        size_text = "{}×{}".format(n, n)
        print("{:<10}{:>16.6f}{:>18.6f}{:>16}".format(size_text, avg_2d, avg_1d, n * n))
    print("-" * 64)
    print("※ 1차원 평균은 보너스 과제(메모리 접근 단순화) 비교용입니다.")


# ---------------------------------------------------------------------------
# 6. 모드 1: 사용자 입력(3×3)
# ---------------------------------------------------------------------------
def read_matrix(name, n=3):
    """n×n 행렬을 한 줄씩(공백 구분) 입력받는다. 형식이 틀리면 그 줄을 다시 입력받는다."""
    print("\n{} ({}×{}) 입력 — 한 줄에 숫자 {}개를 공백으로 구분해 입력하세요.".format(name, n, n, n))
    matrix = create_matrix(n)
    row = 0
    while row < n:
        line = input("  {}행: ".format(row + 1))
        parts = line.split()

        # 열 개수 검증
        if len(parts) != n:
            print("  입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요. "
                  "(현재 {}개)".format(n, len(parts)))
            continue

        # 숫자 파싱 검증
        try:
            values = [float(p) for p in parts]
        except ValueError:
            print("  입력 형식 오류: 숫자만 입력할 수 있습니다. 다시 입력하세요.")
            continue

        for col in range(n):
            set_value(matrix, row, col, values[col])
        row += 1
    return matrix


def print_matrix(matrix):
    for row in matrix:
        print("   " + " ".join("{:>5}".format(v) for v in row))


def run_user_mode():
    print("\n=== 모드 1: 사용자 입력 (3×3) ===")
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")

    print("\n[저장 확인] 필터 A")
    print_matrix(filter_a)
    print("[저장 확인] 필터 B")
    print_matrix(filter_b)

    pattern = read_matrix("패턴")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    time_a = measure_avg_ms(mac, pattern, filter_a)
    time_b = measure_avg_ms(mac, pattern, filter_b)
    result = compare_scores(score_a, score_b, "A", "B", "판정 불가")

    print("\n[결과]")
    print("  필터 A 점수: {}  (연산 시간 {:.6f} ms)".format(score_a, time_a))
    print("  필터 B 점수: {}  (연산 시간 {:.6f} ms)".format(score_b, time_b))
    if result == "판정 불가":
        print("  판정: 판정 불가 (두 점수 차이가 {} 미만 → 동점)".format(EPSILON))
    else:
        print("  판정: {}".format(result))

    print_performance_table([3])


# ---------------------------------------------------------------------------
# 7. 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_json(path):
    """JSON 파일을 읽는다. 실패하면 None과 오류 메시지를 돌려준다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "파일을 찾을 수 없습니다: {}".format(path)
    except json.JSONDecodeError as e:
        return None, "JSON 형식 오류: {}".format(e)


def load_filters(raw_filters):
    """
    filters 섹션을 읽어 {N: {"Cross": 행렬, "X": 행렬}} 형태로 정리한다.
    키 이름('cross', 'x')은 표준 라벨로 정규화한다.
    문제가 있는 필터는 건너뛰고 경고 메시지를 모은다.
    """
    filters = {}
    warnings = []
    if not isinstance(raw_filters, dict):
        warnings.append("filters 섹션이 없거나 형식이 잘못되었습니다.")
        return filters, warnings

    for size_key, group in raw_filters.items():
        # 키 규칙: size_N
        parts = size_key.split("_")
        if len(parts) != 2 or parts[0] != "size" or not parts[1].isdigit():
            warnings.append("필터 키 형식 오류: '{}' (size_N 형태여야 함)".format(size_key))
            continue
        n = int(parts[1])
        if not isinstance(group, dict):
            warnings.append("{}: 필터 묶음 형식 오류".format(size_key))
            continue

        filters[n] = {}
        for raw_label, matrix in group.items():
            label = normalize_label(raw_label)
            if label is None:
                warnings.append("{}: 알 수 없는 필터 라벨 '{}'".format(size_key, raw_label))
                continue
            if not is_square(matrix, n):
                warnings.append("{}/{}: 필터 크기가 {}×{}가 아닙니다.".format(size_key, raw_label, n, n))
                continue
            filters[n][label] = matrix
    return filters, warnings


def parse_pattern_key(key):
    """'size_{N}_{idx}' 키에서 N을 뽑는다. 형식이 틀리면 None."""
    parts = key.split("_")
    if len(parts) != 3 or parts[0] != "size" or not parts[1].isdigit() or not parts[2].isdigit():
        return None
    return int(parts[1])


def analyze_pattern(key, item, filters):
    """
    패턴 하나를 판정한다. 어떤 문제가 생겨도 예외를 밖으로 던지지 않고
    결과 딕셔너리(status, reason 등)로 돌려준다 → 프로그램이 중단되지 않는다.
    """
    result = {"key": key, "cross": None, "x": None, "verdict": "-",
              "expected": "-", "status": "FAIL", "reason": ""}

    # (1) 키에서 N 추출
    n = parse_pattern_key(key)
    if n is None:
        result["reason"] = "스키마 오류: 키가 size_N_idx 형식이 아님"
        return result

    # (2) 해당 크기 필터 선택
    group = filters.get(n)
    if not group or LABEL_CROSS not in group or LABEL_X not in group:
        result["reason"] = "스키마 오류: size_{} 필터(Cross/X)가 없음".format(n)
        return result

    # (3) input / expected 존재 확인
    if not isinstance(item, dict) or "input" not in item or "expected" not in item:
        result["reason"] = "스키마 오류: input 또는 expected 누락"
        return result

    # (4) expected 정규화
    expected = normalize_label(item["expected"])
    if expected is None:
        result["reason"] = "라벨 오류: 알 수 없는 expected 값 '{}'".format(item["expected"])
        return result
    result["expected"] = expected

    # (5) 크기 일치 검증
    pattern = item["input"]
    if not is_square(pattern, n):
        result["reason"] = "크기 불일치: 패턴이 {}×{}가 아님".format(n, n)
        return result

    # (6) MAC 연산 + 판정
    try:
        score_cross = mac(pattern, group[LABEL_CROSS])
        score_x = mac(pattern, group[LABEL_X])
    except TypeError:
        result["reason"] = "데이터 오류: 숫자가 아닌 값이 포함됨"
        return result

    verdict = compare_scores(score_cross, score_x, LABEL_CROSS, LABEL_X, LABEL_UNDECIDED)
    result["cross"] = score_cross
    result["x"] = score_x
    result["verdict"] = verdict

    # (7) PASS / FAIL
    if verdict == expected:
        result["status"] = "PASS"
    elif verdict == LABEL_UNDECIDED:
        result["reason"] = "동점(UNDECIDED): Cross={} X={}, 차이 < {}".format(
            score_cross, score_x, EPSILON)
    else:
        result["reason"] = "판정 불일치: 판정={} expected={}".format(verdict, expected)
    return result


def format_score(value):
    return "-" if value is None else "{:.6f}".format(value)


def run_json_mode():
    print("\n=== 모드 2: data.json 분석 ===")

    # (1) 필터 로드
    data, error = load_json(DATA_FILE)
    if error:
        print("오류:", error)
        return
    if not isinstance(data, dict):
        print("오류: data.json 최상위 구조가 객체가 아닙니다.")
        return

    filters, warnings = load_filters(data.get("filters"))
    print("[필터 로드] 크기: {}".format(", ".join("{}×{}".format(n, n) for n in sorted(filters))))
    for w in warnings:
        print("  경고:", w)

    # (2) 패턴 로드/검증 + MAC + 판정
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("오류: patterns 섹션이 없거나 형식이 잘못되었습니다.")
        return

    results = []
    print()
    print("{:<12}{:>14}{:>14}{:>12}{:>10}{:>7}".format(
        "케이스", "Cross 점수", "X 점수", "판정", "expected", "결과"))
    print("-" * 70)
    for key, item in patterns.items():
        r = analyze_pattern(key, item, filters)
        results.append(r)
        print("{:<12}{:>14}{:>14}{:>12}{:>10}{:>7}".format(
            key, format_score(r["cross"]), format_score(r["x"]),
            r["verdict"], r["expected"], r["status"]))
    print("-" * 70)

    # (3) 성능 분석
    print_performance_table(PERF_SIZES)

    # (4) 결과 요약
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed
    print("\n[결과 요약]")
    print("  전체 테스트: {}  |  통과: {}  |  실패: {}".format(total, passed, failed))
    if failed:
        print("  실패 케이스:")
        for r in results:
            if r["status"] == "FAIL":
                print("   - {}: {}".format(r["key"], r["reason"]))


# ---------------------------------------------------------------------------
# 8. 실행 흐름
# ---------------------------------------------------------------------------
def main():
    while True:
        print("\n========================================")
        print(" MAC 패턴 판별기")
        print("  1. 사용자 입력 (3×3)")
        print("  2. data.json 분석")
        print("  0. 종료")
        print("========================================")
        choice = input("모드를 선택하세요: ").strip()
        if choice == "1":
            run_user_mode()
        elif choice == "2":
            run_json_mode()
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 1, 2, 0 중에서 입력하세요.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n프로그램을 종료합니다.")
