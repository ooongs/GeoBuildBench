# DSL 예제 파일들

이 디렉토리에는 직접 작성하고 테스트해볼 수 있는 DSL 예제 파일들이 있습니다.

## 사용 방법

```bash
# 예제 실행
python generate_from_dsl.py my_examples/01_simple_triangle.txt

# 출력 파일명 지정
python generate_from_dsl.py my_examples/02_circle_and_points.txt my_circle.png
```

## 예제 설명

1. **01_simple_triangle.txt** - 가장 기본적인 삼각형
   - 세 점을 정의하고 선분으로 연결

2. **02_circle_and_points.txt** - 원 그리기
   - 중심점과 반지름으로 원 생성

3. **03_square_with_rotation.txt** - 회전을 이용한 정사각형
   - rotate 명령어로 90도씩 회전
   - 각도를 °(도) 단위로 표현

4. **04_math_expressions.txt** - 수학 표현식 활용
   - 삼각함수와 사칙연산 사용
   - 정육각형 생성 예제

5. **05_perpendicular.txt** - 중점을 이용한 이등분
   - midpoint 명령어로 중점 구하기
   - 교차하는 선분 그리기

## DSL 기본 문법

### 점 정의
```
point : x y -> 점이름
point : 100 150 -> A
```

### 선분
```
segment : 점1 점2 -> 선분이름
segment : A B -> AB
```

### 원
```
circle : 중심점 반지름점 -> 원이름
circle : O A -> circle1
```

### 회전
```
rotate : 점 각도 중심점 -> 새점
rotate : A 90° Center -> B
rotate : A 1.5708rad Center -> B  # 라디안 사용
```

### 중점
```
midpoint : 점1 점2 -> 중점
midpoint : A B -> M
```

### 수직선
```
perpendicular : 점 선분 -> 수직선
perpendicular : M AB -> perp_line
```

### 평행선
```
parallel : 점 선분 -> 평행선
parallel : C AB -> para_line
```

## 주의사항

1. **polygon 명령어는 사용하지 마세요!**
   - 대신 segment로 각 변을 명시적으로 그려야 합니다

2. **각도 표현**
   - 도(degree): `90°` 또는 `90deg`
   - 라디안: `1.5708rad`

3. **수학 표현식**
   - 사칙연산: `100+50`, `200*0.5`
   - 삼각함수: `cos(45°)`, `sin(30°)`, `tan(60°)`

4. **라벨 이름**
   - 영문, 숫자, 한글 모두 사용 가능
   - 공백은 사용할 수 없습니다

## 더 많은 예제

프로젝트의 `examples/` 디렉토리에서 더 많은 예제를 확인할 수 있습니다:

```bash
python preview.py examples/
```
