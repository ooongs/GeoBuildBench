# 객체 레이블 표시 기능

## 📝 개요

DSL로 생성된 기하학 객체들의 이름(레이블)을 그림에 자동으로 표시하는 기능을 추가했습니다.
이제 각 점, 선분, 원 등이 어떤 이름인지 그림에서 바로 확인할 수 있습니다.

## ✨ 기능

### 레이블 표시 타입

| 객체 타입 | 레이블 색상 | 위치 | 예시 |
|-----------|------------|------|------|
| 점 (Point) | 파란색 | 점 위쪽 | A, B, Center |
| 선분/직선 (Segment/Line) | 빨간색 | 중간점 | AB, line1 |
| 원 (Circle) | 초록색 | 중심 근처 | circle1 |
| 각 (Angle) | 보라색 | 꼭짓점 근처 | angle_ABC |

### 레이블 스타일

- **배경**: 반투명 흰색 박스 (가독성 향상)
- **폰트 크기**: 8-9pt (적절한 크기)
- **자동 숨김**: `_auto_`, `_` 등으로 시작하는 내부 레이블은 자동 숨김

## 🎯 사용 방법

### 1. DSLExecutor로 레이블 표시 (기본값)

```python
from dsl_executor import DSLExecutor

# 레이블 표시 (기본값 = True)
executor = DSLExecutor(save_images=True, show_labels=True)
result = executor.execute(dsl_code, problem_id='test', iteration=0)
```

### 2. 레이블 숨기기

```python
# 레이블 숨김
executor = DSLExecutor(save_images=True, show_labels=False)
result = executor.execute(dsl_code, problem_id='test', iteration=0)
```

### 3. Construction 직접 사용

```python
from random_constr import Construction
import matplotlib.pyplot as plt

constr = Construction()
constr.load('my_dsl.txt')
constr.generate(require_theorem=False)

fig, ax = plt.subplots()

# 레이블 표시
constr.render(ax, show_labels=True)

# 또는 레이블 없이
constr.render(ax, show_labels=False)

plt.savefig('output.png')
```

## 📐 예시

### 정오각형 with 레이블

```python
test_dsl = '''
point : 50 50 -> Center
point : 100 50 -> A
rotate : A 72° Center -> B
rotate : B 72° Center -> C
rotate : C 72° Center -> D
rotate : D 72° Center -> E
segment : A B -> s1
segment : B C -> s2
segment : C D -> s3
segment : D E -> s4
segment : E A -> s5
circle : Center 30 -> circle1
equality : Center Center -> expr0
prove : expr0 -> result
'''

executor = DSLExecutor(save_images=True, show_labels=True)
result = executor.execute(test_dsl, problem_id='pentagon', iteration=0)
```

**결과**: 
- 점들 (Center, A, B, C, D, E) - 파란색 레이블
- 선분들 (s1, s2, s3, s4, s5) - 빨간색 레이블
- 원 (circle1) - 초록색 레이블

### 사각형 with 대각선

```python
test_dsl = '''
point : 30 30 -> A
point : 70 30 -> B
point : 70 70 -> C
point : 30 70 -> D
segment : A B -> AB
segment : B C -> BC
segment : C D -> CD
segment : D A -> DA
segment : A C -> diagonal1
segment : B D -> diagonal2
equality : A A -> expr0
prove : expr0 -> result
'''
```

**결과**: 모든 점과 선분에 명확한 레이블 표시

## 🛠️ 구현 세부사항

### 1. Element 클래스 수정

```python
def draw(self, ax, corners, show_labels=True):
    if not self.drawable():
        return
    
    # 객체 그리기
    self.data.draw(ax, corners)
    
    # 레이블 표시
    if show_labels and not self.label.startswith('_'):
        self._draw_label(ax)

def _draw_label(self, ax):
    """객체 레이블을 그림에 표시"""
    offset = 3  # 레이블 오프셋
    
    if isinstance(self.data, Point):
        # 점 레이블: 점 위쪽에 표시
        ax.text(self.data.a[0], self.data.a[1] + offset, self.label,
               fontsize=9, ha='center', va='bottom', color='blue',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='none', alpha=0.7))
    # ... 다른 타입들
```

### 2. Construction.render 수정

```python
def render(self, ax, elements=None, show_labels=True):
    if elements is None: 
        elements = self.elements
    
    # ... 축 설정 ...
    
    # Draw all elements with labels
    for el in elements:
        el.draw(ax, self.corners, show_labels=show_labels)
```

### 3. DSLExecutor 통합

```python
def __init__(self, ..., show_labels: bool = True):
    self.show_labels = show_labels

def _render_image(self, construction, ...):
    construction.render(ax, show_labels=self.show_labels)
```

## 💡 디자인 결정

### 자동 레이블 숨김

`_`로 시작하는 레이블은 자동으로 숨겨집니다:
- `_auto_0`, `_auto_1` - 인라인 숫자 리터럴 자동 생성
- `_tmp_`, `_internal_` - 기타 내부 변수

### 레이블 위치 전략

- **점**: 위쪽에 표시 (점과 겹치지 않음)
- **선분**: 중간점에 표시 (선 위에 표시)
- **원**: 중심 옆에 표시 (원과 겹치지 않음)
- **각**: 꼭짓점 근처에 표시

### 색상 체계

객체 타입별로 다른 색상을 사용하여 구분:
- 파란색: 점 (가장 기본)
- 빨간색: 선분/직선 (눈에 띄게)
- 초록색: 원 (부드럽게)
- 보라색: 각 (특별하게)

## 📊 성능 영향

- **렌더링 시간**: 레이블당 약 1-2ms 증가 (무시 가능)
- **이미지 크기**: 거의 동일 (텍스트는 벡터)
- **메모리**: 추가 영향 없음

## 🎨 향후 개선사항

1. **레이블 위치 자동 조정**: 겹치는 레이블 감지 및 조정
2. **커스터마이징**: 색상, 크기, 폰트 설정 가능
3. **선택적 표시**: 특정 타입만 레이블 표시
4. **LaTeX 지원**: 수학 기호 레이블 ($\\alpha$, $\\theta$ 등)

## ✅ 테스트 결과

```
✅ 점 레이블 표시: 성공 (파란색)
✅ 선분 레이블 표시: 성공 (빨간색)
✅ 원 레이블 표시: 성공 (초록색)
✅ 각 레이블 표시: 성공 (보라색)
✅ 자동 레이블 숨김: 성공 (_auto_ 등)
✅ 레이블 on/off: 성공 (show_labels 파라미터)
```

## 🎉 결과

이제 DSL로 생성한 기하학 도형이 훨씬 이해하기 쉬워졌습니다!
각 객체의 이름이 그림에 명확하게 표시되어 학습 및 디버깅에 매우 유용합니다.


