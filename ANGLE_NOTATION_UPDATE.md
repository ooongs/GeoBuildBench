# 각도 표기법 업데이트

## 📝 개요

DSL에서 `°` 기호를 사용한 degree 표기와 `rad`/`r`을 사용한 radian 표기를 지원합니다.
이제 각도를 명시적으로 표현할 수 있어 코드 가독성이 크게 향상되었습니다.

## ✨ 새로운 기능

### 이전 방식
```
const int 90 -> deg
rotate : P deg Center -> Q
```

### 새로운 방식
```
# Degree (도)
rotate : P 90° Center -> Q

# Radian (라디안)
rotate : P 1.5708rad Center -> Q

# 짧은 표기
rotate : P 1.5708r Center -> Q

# 기본 (degree, 하위 호환성)
rotate : P 90 Center -> Q
```

## 📐 지원하는 각도 표기

| 표기법 | 예시 | 의미 | 내부 처리 |
|--------|------|------|-----------|
| `°` | `90°` | 90도 | AngleSize(π/2 rad) |
| `rad` | `1.5708rad` | 1.5708 라디안 | AngleSize(1.5708 rad) |
| `r` | `1.5708r` | 1.5708 라디안 (짧은 표기) | AngleSize(1.5708 rad) |
| 일반 숫자 | `90` | 90도 (기본값) | int → degree |

## 🔍 정밀도 검증

실제 회전 각도 정밀도 테스트 결과:

```
✅ 90° (degree)     : 예상 1.5708rad, 실제 1.5708rad (오차 0.000000)
✅ 45° (degree)     : 예상 0.7854rad, 실제 0.7854rad (오차 0.000000)
✅ 1.5708rad        : 예상 1.5708rad, 실제 1.5708rad (오차 0.000000)
✅ 90 (일반 숫자)    : 예상 1.5708rad, 실제 1.5708rad (오차 0.000000)
```

## 🛠️ 구현 세부사항

### 1. 파싱 로직 (`random_constr.py`)

```python
# 각도 표기 확인
is_degree = label.endswith('°')
is_radian = label.endswith('rad') or label.endswith('r')

# 각도 기호 제거 후 숫자 추출
if is_degree:
    clean_label = label[:-1]  # ° 제거
elif is_radian:
    if label.endswith('rad'):
        clean_label = label[:-3]  # rad 제거
    else:
        clean_label = label[:-1]  # r 제거

# 타입에 따라 변환
if is_radian:
    element.data = AngleSize(value)  # 그대로 radian
elif is_degree:
    element.data = AngleSize(np.radians(value))  # degree → radian 변환
```

### 2. 타입 매핑

- `°` → `AngleSize` 객체 (radian 값 저장)
- `rad`/`r` → `AngleSize` 객체 (radian 값 저장)
- 일반 숫자 → `int`/`float` (기존 동작 유지)

### 3. 함수 호출

- `AngleSize` → `rotate_pAp(point, angle_size, by_point)` 호출
- `int` → `rotate_pip(point, degrees, by_point)` 호출 (degree로 해석)

## 🎯 사용 예시

### 정삼각형 (120° 회전)
```
point : 50 50 -> Center
point : 100 50 -> P0
rotate : P0 120° Center -> P1
rotate : P1 120° Center -> P2
segment : P0 P1 -> s1
segment : P1 P2 -> s2
segment : P2 P0 -> s3
equality : Center Center -> expr0
prove : expr0 -> result
```

### 정사각형 (90° 회전)
```
point : 50 50 -> Center
point : 100 50 -> P0
rotate : P0 90° Center -> P1
rotate : P1 90° Center -> P2
rotate : P2 90° Center -> P3
polygon : P0 P1 P2 P3 -> square s0 s1 s2 s3
equality : Center Center -> expr0
prove : expr0 -> result
```

### Radian 사용 (π/6 회전)
```
point : 50 50 -> Center
point : 100 50 -> P
rotate : P 0.5236rad Center -> Q  # π/6 ≈ 30°
segment : Center P -> s1
segment : Center Q -> s2
equality : Center Center -> expr0
prove : expr0 -> result
```

### 소수점 각도
```
point : 50 50 -> Center
point : 100 50 -> P
rotate : P 45.5° Center -> Q     # 45.5도
rotate : Q 22.3° Center -> R     # 22.3도
equality : Center Center -> expr0
prove : expr0 -> result
```

## 💡 장점

1. **가독성 향상**: `90°`가 `const int 90 -> deg`보다 직관적
2. **명시적 단위**: degree와 radian을 명확히 구분
3. **하위 호환성**: 기존 코드 그대로 작동 (일반 숫자 = degree)
4. **정밀도 보존**: 소수점 각도 지원 (`45.5°`, `1.5708rad`)
5. **간결성**: 코드가 짧고 명확함

## 📚 문서 업데이트

- `prompts/dsl_guidelines.txt` - 각도 표기 예시 추가
- `prompts/system_prompt.txt` - 회전 명령어 업데이트
- `random_constr.py` - 각도 파싱 로직 추가

## 🔄 하위 호환성

기존 코드는 모두 정상 작동:

```
# ✅ 기존 방식 (여전히 작동)
const int 90 -> deg
rotate : P deg Center -> Q

# ✅ 일반 숫자 (여전히 degree로 해석)
rotate : P 90 Center -> Q

# ✅ 새로운 방식 (더 명확)
rotate : P 90° Center -> Q
```

## 🚀 성능

- 파싱 단계에서 문자열 체크 추가 (`.endswith()`)
- 성능 영향 미미 (O(1) 문자열 체크)
- 실행 성능 동일 (AngleSize 객체 사용)

## 🎉 결과

모든 각도 표기법이 정확하게 작동하며, 코드 가독성이 크게 향상되었습니다!

```
✅ degree 기호 (90°)         : 성공
✅ radian 표기 (1.5708rad)   : 성공
✅ 짧은 radian (1.5708r)     : 성공
✅ 일반 숫자 (90)            : 성공
✅ 소수점 각도 (45.5°)       : 성공
```


