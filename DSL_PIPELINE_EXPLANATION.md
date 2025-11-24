# DSL 파이프라인 상세 설명

## 📝 텍스트 → 그림 변환 과정

### 예제 DSL 코드:
```txt
const int 150 -> cx
const int 150 -> cy
point : cx cy -> Center
const int 90 -> deg
rotate : Center deg Center -> P1
segment : Center P1 -> s1
equality : Center Center -> expr0
prove : expr0 -> result
```

---

## 1단계: 텍스트 파싱 (random_constr.py)

### 파일 읽기
```python
def load(self, filename):
    with open(filename, 'r') as f:
        for line in f:
            command = parse_command(line, self.element_dict)
```

### parse_command 함수
```python
def parse_command(line, element_dict):
    tokens = line.split()  # 공백으로 분리
    
    # 예: "const int 150 -> cx"
    # tokens = ['const', 'int', '150', '->', 'cx']
    
    if tokens[0] == "const":
        # 상수 명령어
        datatype = str_to_const_type[tokens[1]]  # int
        value = float(tokens[2])                  # 150.0
        label = tokens[4]                         # 'cx'
        
        element = Element(label, element_dict)
        command = ConstCommand(datatype, value, element)
        return command
    
    else:
        # 일반 명령어
        # 예: "point : cx cy -> Center"
        command_name = tokens[0]  # 'point'
        
        # ':' 이후부터 '->' 앞까지가 입력
        # '->' 이후가 출력
        labels = tokens[2:]  # ['cx', 'cy', '->', 'Center']
        arrow_index = labels.index("->")
        
        input_labels = labels[:arrow_index]    # ['cx', 'cy']
        output_labels = labels[arrow_index+1:]  # ['Center']
        
        # 입력 Element 찾기
        input_elements = [element_dict[label] for label in input_labels]
        
        # 출력 Element 생성
        output_elements = [Element(label, element_dict) 
                          for label in output_labels]
        
        command = Command(command_name, input_elements, output_elements)
        return command
```

---

## 2단계: Element와 Command 객체 생성

### Element 클래스
```python
class Element:
    def __init__(self, label, element_dict):
        self.data = None        # 실제 기하 객체가 여기 저장됨
        self.label = label      # 'Center', 'P1' 등
        element_dict[label] = self
```

### Command 클래스
```python
class Command:
    def __init__(self, command_name, input_elements, output_elements):
        self.name = command_name           # 'point', 'rotate' 등
        self.input_elements = input_elements
        self.output_elements = output_elements
```

### 예제 실행 결과:
```
파싱 후 상태:
- element_dict = {
    'cx': Element(label='cx', data=150),
    'cy': Element(label='cy', data=150),
    'Center': Element(label='Center', data=None),  # 아직 실행 안됨
    ...
  }
- nc_commands = [
    Command('point', [cx, cy], [Center]),
    Command('rotate', [Center, deg, Center], [P1]),
    ...
  ]
```

---

## 3단계: 명령어 실행 (commands.py)

### generate() 호출 시:
```python
def generate(self):
    self.run_commands()
    self.fit_to_window()

def run_commands(self):
    for command in self.nc_commands:
        command.apply()
```

### Command.apply() 실행:
```python
def apply(self):
    # 1. 입력 Element에서 실제 데이터 가져오기
    input_data = [el.data for el in self.input_elements]
    # 예: [150, 150] (cx.data, cy.data)
    
    # 2. 명령어 이름과 타입으로 함수 찾기
    name = command_types_name(self.name, input_data)
    # 예: "point_ii" (point + int + int)
    
    f = command_dict[name]  # commands.py에서 함수 가져오기
    
    # 3. 함수 실행
    output_data = f(*input_data)
    # 예: point_ii(150, 150) → Point([150, 150])
    
    # 4. 결과를 출력 Element에 저장
    for data, element in zip(output_data, self.output_elements):
        element.data = data
```

### 타입 매칭 시스템:
```python
# type_to_shortcut 딕셔너리
{
    int: 'i',
    Point: 'p',
    Measure: 'm',
    AngleSize: 'A',
    ...
}

# command_types_name 함수
def command_types_name(name, params):
    # 예: name='point', params=[150, 150]
    # 타입: [int, int]
    # 결과: 'point_ii'
    return "{}_{}".format(
        name, 
        ''.join(type_to_shortcut[type(x)] for x in params)
    )
```

---

## 4단계: 기하 객체 생성 (geo_types.py)

### commands.py의 함수들:
```python
def point_ii(x, y):
    """고정된 좌표로 점 생성"""
    return gt.Point([float(x), float(y)])

def rotate_pip(point, degrees, by_point):
    """각도로 회전"""
    radians = np.radians(float(degrees))
    return gt.Point(by_point.a + gt.rotate_vec(point.a - by_point.a, radians))

def segment_pp(p1, p2):
    """두 점으로 선분 생성"""
    return gt.Segment(p1.a, p2.a)
```

### geo_types.py의 클래스들:
```python
class Point:
    def __init__(self, a):
        self.a = np.array(a, dtype=float)  # [x, y] 좌표
    
    def draw(self, ax, corners):
        ax.plot(self.a[0], self.a[1], 'ko', markersize=5)

class Segment(Line):
    def __init__(self, p1, p2):
        # 두 점으로 선분 정의
        self.end_points = np.array([p1, p2])
    
    def draw(self, ax, corners):
        ax.plot([self.end_points[0][0], self.end_points[1][0]], 
                [self.end_points[0][1], self.end_points[1][1]], 
                'k-', linewidth=1)
```

---

## 5단계: 렌더링 (matplotlib)

### render() 함수:
```python
def render(self, ax, elements=None):
    if elements is None:
        elements = self.elements
    
    # 1. axes 설정
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(self.corners[0][0], self.corners[1][0])
    ax.set_ylim(self.corners[0][1], self.corners[1][1])
    ax.axis('off')
    
    # 2. 모든 요소 그리기
    for element in elements:
        element.draw(ax, self.corners)
```

### Element.draw():
```python
def draw(self, ax, corners):
    if self.drawable():
        self.data.draw(ax, corners)
        # Point, Line, Segment, Circle 등의 draw() 메서드 호출
```

---

## 📊 전체 흐름도

```
텍스트 파일 (DSL)
    ↓
[1. 파싱]
parse_command()
    ↓
Element + Command 객체들
    ↓
[2. 실행]
command.apply()
    ↓
commands.py 함수 호출
(point_ii, rotate_pip, ...)
    ↓
[3. 기하 객체 생성]
geo_types.py 클래스 인스턴스
(Point, Segment, Circle, ...)
    ↓
[4. 렌더링]
element.draw(ax, corners)
    ↓
matplotlib Figure
    ↓
PNG 이미지
```

---

## 🔍 상세 예제 실행

### DSL 코드:
```txt
const int 150 -> cx
const int 150 -> cy
point : cx cy -> Center
```

### 실행 과정:

#### Step 1: 파싱
```python
# Line 1: "const int 150 -> cx"
cx = Element('cx', element_dict)
ConstCommand(int, 150, cx).apply()
→ cx.data = 150

# Line 2: "const int 150 -> cy"
cy = Element('cy', element_dict)
ConstCommand(int, 150, cy).apply()
→ cy.data = 150

# Line 3: "point : cx cy -> Center"
Center = Element('Center', element_dict)
cmd = Command('point', [cx, cy], [Center])
```

#### Step 2: 실행
```python
cmd.apply():
    input_data = [cx.data, cy.data]  # [150, 150]
    
    # 타입 확인: [int, int]
    name = "point_ii"
    
    # 함수 호출
    result = point_ii(150, 150)
    
    # point_ii 내부:
    return Point([float(150), float(150)])
    
    # 결과 저장
    Center.data = Point([150.0, 150.0])
```

#### Step 3: 객체 생성
```python
Point.__init__([150, 150]):
    self.a = np.array([150.0, 150.0])
```

#### Step 4: 렌더링
```python
Center.draw(ax, corners):
    Center.data.draw(ax, corners)
    
Point.draw(ax, corners):
    ax.plot(150.0, 150.0, 'ko', markersize=5)
```

---

## 💡 핵심 포인트

1. **타입 기반 디스패치**: 입력 타입에 따라 다른 함수 호출
   - `point : int int` → `point_ii()`
   - `point : Measure Measure` → `point_mm()`

2. **지연 실행**: 파싱과 실행이 분리됨
   - 먼저 모든 명령어 파싱
   - 나중에 순차적으로 실행

3. **Element는 컨테이너**: label과 data를 연결
   - DSL의 변수 = Element
   - Element.data = 실제 기하 객체

4. **명령어 = 함수 호출**: 각 DSL 명령어는 Python 함수에 매핑됨

---

## 🎯 확장 방법

새로운 명령어 추가하기:

### 1. commands.py에 함수 추가:
```python
def my_command_pp(point1, point2):
    """두 점으로 무언가 하기"""
    result = ...  # 계산
    return result
```

### 2. DSL에서 사용:
```txt
my_command : P1 P2 -> Result
```

시스템이 자동으로 `my_command_pp` 함수를 찾아서 실행합니다!




