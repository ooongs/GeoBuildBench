# 숫자 리터럴 지원 업데이트

## 📝 개요

DSL에서 인라인 숫자 리터럴을 지원하도록 파서를 업데이트했습니다. 이제 `const` 선언 없이도 숫자를 직접 사용할 수 있습니다.

## ✨ 새로운 기능

### 이전 방식 (장황함)
```
const int 0 -> x
const int 0 -> y
point : x y -> A

const int 50 -> radius
circle : O radius -> circle1
```

### 새로운 방식 (간결함!)
```
point : 0 0 -> A
circle : O 50 -> circle1
```

## 🔍 정밀도 보존

### 정수 리터럴
- `100` → `int` 타입으로 저장
- 예: `point : 100 200 -> A`

### 소수점 리터럴
- `50.5` → `float` 타입으로 저장 (정밀도 보존!)
- 예: `point : 50.5 86.7 -> B`

### 혼합 사용
```
point : 0 0 -> A           # int, int
point : 100.5 200 -> B     # float, int
point : 50 86.7 -> C       # int, float
polygon : A B C -> tri c a b
```

## 🛠️ 구현 세부사항

### 1. `random_constr.py` 수정

#### `type_to_shortcut` 딕셔너리
```python
type_to_shortcut = {
    int       : 'i',
    float     : 'i',  # float를 int와 동일하게 처리
    Boolean   : 'b',
    Measure   : 'm',
    ...
}
```

#### `parse_command` 함수
- 숫자 리터럴 자동 감지
- 자동 레이블 생성 (`_auto_0`, `_auto_1`, ...)
- 소수점 유무에 따라 int/float 타입 결정
- element.data에 직접 값 설정 (정밀도 보존)

### 2. 타입 호환성

`commands.py`의 함수들이 int와 float를 모두 받을 수 있도록 설계되어 있음:

```python
def point_ii(x, y):
    """고정된 좌표 (x, y)에 점 생성"""
    return gt.Point([float(x), float(y)])  # 내부에서 float()로 변환
```

따라서 `point_ii`는 `(int, int)`, `(float, float)`, `(int, float)` 모두 처리 가능.

## ✅ 테스트 결과

모든 테스트 통과:
- ✅ 정수 좌표
- ✅ 소수점 좌표
- ✅ 혼합 좌표
- ✅ 소수점 반지름
- ✅ 소수점 각도

## 📚 문서 업데이트

- `prompts/dsl_guidelines.txt` - 인라인 리터럴 예시 추가
- `prompts/system_prompt.txt` - 새 문법 설명 추가
- `dsl_executor.py` - 테스트 코드 업데이트

## 🎯 사용 예시

### 좌표 지정
```
point : 0 0 -> Origin
point : 100.5 200.3 -> P
```

### 각도 회전
```
point :  -> Center
point :  -> P
rotate : P 45.5 Center -> Q    # 45.5도 회전
```

### 원 반지름
```
point : 50 50 -> O
circle : O 30.5 -> circle1     # 반지름 30.5
```

## 💡 주의사항

1. **Measure 타입이 필요한 경우**: 명시적으로 `const Measure X -> label` 사용
2. **자동 생성 레이블**: `_auto_` 접두사는 시스템 예약어
3. **기존 호환성**: 기존 `const` 문법도 계속 사용 가능

## 🚀 성능 영향

- 파싱 단계에서 약간의 오버헤드 (숫자 감지)
- 실행 성능에는 영향 없음
- 메모리 사용량 증가 미미 (자동 element 생성)


